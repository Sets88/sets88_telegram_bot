"""
Quiz Module - Core Business Logic
Builds multiple-choice quizzes out of user-supplied Markdown material via LLM,
stores them as JSON on disk and serves randomized practice sessions.
"""

import os
import json
import uuid
import asyncio
import random
import re
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Optional, List, Dict, Any

from logger import logger
from lib.llm import openrouter_instance


DEFAULT_MODEL = "openai/gpt-5-mini"

MAX_SOURCE_CHARS = 150_000
CHUNK_TARGET_CHARS = 2_500
CHARS_PER_QUESTION = 150
MIN_QUESTIONS = 20
MAX_QUESTIONS = 1_200
MAX_QUESTIONS_PER_CHUNK = 20  # keep a single LLM response short enough to stay well-formed
MIN_OPTIONS = 4
MAX_OPTIONS = 6
SESSION_SIZE = 20
GENERATION_CONCURRENCY = 8

# Telegram language_code -> (human-readable name, writing system)
LANGUAGES: Dict[str, tuple] = {
    'ru': ('Russian', 'cyrillic'),
    'uk': ('Ukrainian', 'cyrillic'),
    'be': ('Belarusian', 'cyrillic'),
    'bg': ('Bulgarian', 'cyrillic'),
    'sr': ('Serbian', 'cyrillic'),
    'el': ('Greek', 'greek'),
    'en': ('English', 'latin'),
    'de': ('German', 'latin'),
    'fr': ('French', 'latin'),
    'es': ('Spanish', 'latin'),
    'it': ('Italian', 'latin'),
    'pt': ('Portuguese', 'latin'),
    'pl': ('Polish', 'latin'),
    'tr': ('Turkish', 'latin'),
}
DEFAULT_LANGUAGE = 'ru'


@dataclass
class QuizQuestion:
    """A single multiple-choice question, optionally mirrored in a second language"""
    id: str
    question: str
    options: List[str]
    correct_index: int
    explanation: str
    # Translation into the quiz's target language. `options_translated` is index-aligned
    # with `options`, so shuffling must permute both together.
    question_translated: Optional[str] = None
    options_translated: Optional[List[str]] = None
    explanation_translated: Optional[str] = None

    @property
    def has_translation(self) -> bool:
        return bool(self.question_translated and self.options_translated)

    @staticmethod
    def create(
        question: str,
        options: List[str],
        correct_index: int,
        explanation: str,
        question_translated: Optional[str] = None,
        options_translated: Optional[List[str]] = None,
        explanation_translated: Optional[str] = None,
    ) -> "QuizQuestion":
        return QuizQuestion(
            id=str(uuid.uuid4()),
            question=question,
            options=options,
            correct_index=correct_index,
            explanation=explanation,
            question_translated=question_translated,
            options_translated=options_translated,
            explanation_translated=explanation_translated
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "QuizQuestion":
        known = {f for f in QuizQuestion.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in known}
        return QuizQuestion(**filtered)


@dataclass
class Quiz:
    """A quiz generated from a Markdown document"""
    id: str
    title: str
    created_at: str
    status: str  # 'generating' | 'ready' | 'failed'
    source_chars: int
    chunks_total: int
    chunks_done: int = 0
    error: Optional[str] = None
    questions: List[QuizQuestion] = field(default_factory=list)
    attempts: int = 0
    last_score: Optional[int] = None
    last_total: Optional[int] = None
    best_score: Optional[int] = None
    source_language: Optional[str] = None   # human-readable, e.g. "Greek"
    target_language: Optional[str] = None   # None when the quiz is monolingual

    @property
    def has_translation(self) -> bool:
        return bool(self.target_language) and any(q.has_translation for q in self.questions)

    @staticmethod
    def create(
        title: str,
        source_chars: int,
        chunks_total: int,
        source_language: Optional[str] = None,
        target_language: Optional[str] = None,
    ) -> "Quiz":
        return Quiz(
            id=str(uuid.uuid4()),
            title=title,
            created_at=datetime.utcnow().isoformat() + "Z",
            status='generating',
            source_chars=source_chars,
            chunks_total=chunks_total,
            source_language=source_language,
            target_language=target_language
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def meta_dict(self) -> Dict[str, Any]:
        """Quiz metadata without the question bank (used by list/status endpoints)"""
        data = asdict(self)
        data.pop('questions', None)
        data['question_count'] = len(self.questions)
        data['has_translation'] = self.has_translation
        return data

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Quiz":
        known = {f for f in Quiz.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in known}
        filtered['questions'] = [QuizQuestion.from_dict(q) for q in data.get('questions', [])]
        return Quiz(**filtered)


# ========== Pure Helpers ==========

def split_markdown(content: str) -> List[str]:
    """
    Split markdown into chunks of roughly CHUNK_TARGET_CHARS characters.
    Prefers heading boundaries, then paragraph boundaries, then a raw slice.
    """
    blocks = [b for b in re.split(r'(?m)^(?=#{1,3} )', content) if b.strip()]
    if not blocks:
        return []

    # Hard-split any block that is bigger than the target on its own
    sized_blocks: List[str] = []
    for block in blocks:
        if len(block) <= CHUNK_TARGET_CHARS:
            sized_blocks.append(block)
            continue

        current = ''
        for paragraph in block.split('\n\n'):
            piece = paragraph if not current else f'{current}\n\n{paragraph}'
            if len(piece) <= CHUNK_TARGET_CHARS:
                current = piece
                continue

            if current:
                sized_blocks.append(current)
                current = ''

            # A single paragraph that still does not fit - slice it raw
            while len(paragraph) > CHUNK_TARGET_CHARS:
                sized_blocks.append(paragraph[:CHUNK_TARGET_CHARS])
                paragraph = paragraph[CHUNK_TARGET_CHARS:]
            current = paragraph

        if current:
            sized_blocks.append(current)

    # Greedily merge neighbouring blocks up to the target size
    chunks: List[str] = []
    current = ''
    for block in sized_blocks:
        piece = block if not current else f'{current}\n\n{block}'
        if len(piece) <= CHUNK_TARGET_CHARS:
            current = piece
        else:
            if current:
                chunks.append(current)
            current = block

    if current:
        chunks.append(current)

    return [c for c in chunks if c.strip()]


def plan_question_counts(chunks: List[str]) -> List[int]:
    """
    Decide how many questions each chunk should produce.
    Total scales with the size of the material, clamped to [MIN_QUESTIONS, MAX_QUESTIONS].
    """
    if not chunks:
        return []

    total_chars = sum(len(c) for c in chunks)
    total = round(total_chars / CHARS_PER_QUESTION)
    total = max(MIN_QUESTIONS, min(MAX_QUESTIONS, total))
    total = max(total, len(chunks))  # at least one question per chunk
    total = min(total, len(chunks) * MAX_QUESTIONS_PER_CHUNK)

    counts = [
        max(1, min(MAX_QUESTIONS_PER_CHUNK, round(total * len(c) / total_chars)))
        for c in chunks
    ]

    # Trim/pad so the counts sum exactly to `total`, respecting the per-chunk ceiling
    while sum(counts) > total:
        idx = counts.index(max(counts))
        if counts[idx] <= 1:
            break
        counts[idx] -= 1
    while sum(counts) < total:
        idx = min(range(len(counts)), key=lambda i: counts[i])
        if counts[idx] >= MAX_QUESTIONS_PER_CHUNK:
            break
        counts[idx] += 1

    return counts


def derive_title(content: str) -> str:
    """Derive a quiz title from the material: first H1, else first non-empty line"""
    heading = re.search(r'(?m)^#\s+(.+)$', content)
    if heading:
        return heading.group(1).strip()[:80]

    for line in content.splitlines():
        stripped = line.strip().lstrip('#').strip()
        if stripped:
            return stripped[:80]

    return f"Quiz {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}"


# Question banks get large, and the webapp polls the list endpoint every few seconds while
# generating. Cache parsed metadata per file, keyed on mtime+size so edits invalidate it.
_META_CACHE: Dict[str, Any] = {}


def normalize_question(text: str) -> str:
    """Normalized form of a question, used to drop duplicates across chunks"""
    return re.sub(r'\W+', ' ', text.lower()).strip()


def detect_script(text: str) -> Optional[str]:
    """
    Detect the dominant writing system of a text: 'greek', 'cyrillic' or 'latin'.
    Cheap and deterministic - enough to tell whether a translation is worth generating.
    """
    counts = {
        'greek': len(re.findall(r'[Ͱ-Ͽἀ-῿]', text)),
        'cyrillic': len(re.findall(r'[Ѐ-ӿ]', text)),
        'latin': len(re.findall(r'[A-Za-z]', text)),
    }

    total = sum(counts.values())
    if total < 20:
        return None

    script, count = max(counts.items(), key=lambda kv: kv[1])
    return script if count / total > 0.5 else None


def resolve_language(code: Optional[str]) -> tuple:
    """Map a Telegram language_code to (name, script). Unknown codes keep their own name."""
    code = (code or '').split('-')[0].lower()

    if code in LANGUAGES:
        return LANGUAGES[code]
    if code:
        return (code, None)

    return LANGUAGES[DEFAULT_LANGUAGE]


def needs_translation(content: str, target_code: Optional[str]) -> bool:
    """False when the material is already written in the target language"""
    if not target_code:
        return False

    _, target_script = resolve_language(target_code)
    source_script = detect_script(content)

    if source_script is None or target_script is None:
        return True  # can't tell - translating is the safe default

    return source_script != target_script


def validate_question(raw: Dict[str, Any]) -> Optional[QuizQuestion]:
    """Validate one LLM-produced question. Returns None (and logs) if it is malformed."""
    try:
        question = str(raw.get('question', '')).strip()
        explanation = str(raw.get('explanation', '')).strip()
        options = [str(o).strip() for o in raw.get('options', []) if str(o).strip()]
        correct_index = int(raw.get('correct_index', -1))
    except (TypeError, ValueError) as exc:
        logger.error(f"Malformed quiz question: {exc}: {raw}")
        return None

    if not question or not explanation:
        logger.error(f"Quiz question missing text or explanation: {raw}")
        return None

    if not (MIN_OPTIONS <= len(options) <= MAX_OPTIONS):
        logger.error(f"Quiz question has {len(options)} options, expected {MIN_OPTIONS}-{MAX_OPTIONS}: {raw}")
        return None

    if len({o.lower() for o in options}) != len(options):
        logger.error(f"Quiz question has duplicate options: {raw}")
        return None

    if not (0 <= correct_index < len(options)):
        logger.error(f"Quiz question correct_index {correct_index} out of range: {raw}")
        return None

    # The translation is optional: a bad one is dropped, the question itself still counts
    q_tr = str(raw.get('question_translated') or '').strip() or None
    e_tr = str(raw.get('explanation_translated') or '').strip() or None
    o_tr = [str(o).strip() for o in (raw.get('options_translated') or [])]

    if len(o_tr) != len(options) or not all(o_tr) or not q_tr:
        if q_tr or o_tr:
            logger.error(f"Dropping malformed translation for question: {question!r}")
        q_tr, o_tr, e_tr = None, None, None

    return QuizQuestion.create(question, options, correct_index, explanation, q_tr, o_tr or None, e_tr)


class QuizManager:
    """
    Manages quizzes for a user: JSON storage, LLM generation, practice sessions
    """

    def __init__(self, user_id: int):
        self.user_id = user_id
        self._lock = asyncio.Lock()  # For concurrent file write protection

    # ========== File Path Helpers ==========

    def _get_quizzes_dir(self) -> str:
        """Get path to the user's quiz directory"""
        dirname = os.path.basename(f'{self.user_id}')
        return os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', 'greek_quizzes', dirname)
        )

    def _get_quiz_path(self, quiz_id: str) -> Optional[str]:
        """Get path to a single quiz file, or None if quiz_id is not a valid UUID"""
        try:
            uuid.UUID(str(quiz_id))
        except (ValueError, AttributeError, TypeError):
            logger.error(f"Invalid quiz id requested by user {self.user_id}: {quiz_id!r}")
            return None

        filename = os.path.basename(f'{quiz_id}.json')
        return os.path.join(self._get_quizzes_dir(), filename)

    # ========== JSON Load/Save Operations ==========

    async def list_quizzes(self) -> List[Dict[str, Any]]:
        """List quiz metadata (no questions), newest first"""
        directory = self._get_quizzes_dir()

        if not os.path.isdir(directory):
            return []

        metas = []
        for filename in os.listdir(directory):
            if not filename.endswith('.json'):
                continue

            path = os.path.join(directory, filename)
            try:
                stat = os.stat(path)
                stamp = (stat.st_mtime_ns, stat.st_size)

                cached = _META_CACHE.get(path)
                if cached and cached[0] == stamp:
                    metas.append(cached[1])
                    continue

                with open(path, 'r', encoding='utf-8') as f:
                    meta = Quiz.from_dict(json.load(f)).meta_dict()

                _META_CACHE[path] = (stamp, meta)
                metas.append(meta)
            except (OSError, json.JSONDecodeError, TypeError, KeyError) as exc:
                logger.error(f"Error loading quiz {path} for user {self.user_id}: {exc}")

        metas.sort(key=lambda m: m.get('created_at', ''), reverse=True)
        return metas

    async def load_quiz(self, quiz_id: str) -> Optional[Quiz]:
        """Load a single quiz from disk"""
        path = self._get_quiz_path(quiz_id)

        if not path or not os.path.exists(path):
            return None

        try:
            with open(path, 'r', encoding='utf-8') as f:
                return Quiz.from_dict(json.load(f))
        except (FileNotFoundError, json.JSONDecodeError, TypeError, KeyError) as exc:
            logger.error(f"Error loading quiz {quiz_id} for user {self.user_id}: {exc}")
            return None

    async def save_quiz(self, quiz: Quiz) -> None:
        """Save a quiz to disk"""
        path = self._get_quiz_path(quiz.id)
        if not path:
            return

        async with self._lock:
            os.makedirs(os.path.dirname(path), exist_ok=True)

            try:
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(quiz.to_dict(), f, ensure_ascii=False, indent=2)
            except Exception as exc:
                logger.error(f"Error saving quiz {quiz.id} for user {self.user_id}: {exc}")

    async def delete_quiz(self, quiz_id: str) -> bool:
        """Delete a quiz file. Returns True if it existed."""
        path = self._get_quiz_path(quiz_id)

        if not path or not os.path.exists(path):
            return False

        try:
            os.remove(path)
            _META_CACHE.pop(path, None)
            logger.info(f"Deleted quiz {quiz_id} for user {self.user_id}")
            return True
        except OSError as exc:
            logger.error(f"Error deleting quiz {quiz_id} for user {self.user_id}: {exc}")
            return False

    # ========== Quiz Creation & Generation ==========

    async def create_quiz(self, title: str, content: str, target_code: Optional[str] = None) -> Quiz:
        """
        Create a quiz record in 'generating' state. Raises ValueError on invalid material.
        `target_code` is a language code to mirror the questions into ('' / None = monolingual).
        The caller is responsible for scheduling `generate()`.
        """
        content = (content or '').strip()

        if not content:
            raise ValueError('Material is empty')

        if len(content) > MAX_SOURCE_CHARS:
            raise ValueError(f'Material is too large: {len(content)} chars, maximum is {MAX_SOURCE_CHARS}')

        chunks = split_markdown(content)
        if not chunks:
            raise ValueError('Could not split the material into anything usable')

        # Only translate when the material is actually in a different writing system
        target_language = None
        if needs_translation(content, target_code):
            target_language, _ = resolve_language(target_code)

        source_script = detect_script(content)

        quiz = Quiz.create(
            title=(title or '').strip() or derive_title(content),
            source_chars=len(content),
            chunks_total=len(chunks),
            source_language=source_script,
            target_language=target_language
        )
        await self.save_quiz(quiz)

        logger.info(
            f"Created quiz {quiz.id} ({quiz.title!r}) for user {self.user_id}: "
            f"{len(content)} chars, {len(chunks)} chunks, script={source_script}, "
            f"translate_to={target_language or 'none'}"
        )
        return quiz

    async def generate(self, quiz_id: str, content: str) -> None:
        """
        Background worker: generate questions chunk by chunk, persisting progress after each one.
        Aborts silently if the quiz file disappears (user deleted it mid-generation).
        """
        try:
            quiz = await self.load_quiz(quiz_id)
            target_language = quiz.target_language if quiz else None

            chunks = split_markdown(content)
            counts = plan_question_counts(chunks)
            semaphore = asyncio.Semaphore(GENERATION_CONCURRENCY)

            async def worker(chunk: str, count: int) -> List[QuizQuestion]:
                async with semaphore:
                    return await self._generate_chunk(chunk, count, target_language)

            tasks = [asyncio.create_task(worker(c, n)) for c, n in zip(chunks, counts)]

            aborted = False
            for future in asyncio.as_completed(tasks):
                questions = await future

                quiz = await self.load_quiz(quiz_id)
                if quiz is None:
                    logger.info(f"Quiz {quiz_id} disappeared during generation, aborting")
                    aborted = True
                    break

                # Dense generation makes near-duplicates likely across chunks
                seen = {normalize_question(q.question) for q in quiz.questions}
                for question in questions:
                    key = normalize_question(question.question)
                    if key in seen:
                        continue
                    seen.add(key)
                    quiz.questions.append(question)

                quiz.chunks_done += 1
                await self.save_quiz(quiz)

            if aborted:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                return

            quiz = await self.load_quiz(quiz_id)
            if quiz is None:
                return

            if quiz.questions:
                quiz.status = 'ready'
            else:
                quiz.status = 'failed'
                quiz.error = 'The model did not return any usable questions'

            await self.save_quiz(quiz)
            logger.info(f"Quiz {quiz_id} finished with status {quiz.status}, {len(quiz.questions)} questions")

        except Exception as exc:
            logger.exception(f"Error generating quiz {quiz_id} for user {self.user_id}: {exc}")
            quiz = await self.load_quiz(quiz_id)
            if quiz is not None:
                quiz.status = 'failed'
                quiz.error = str(exc)
                await self.save_quiz(quiz)

    async def _generate_chunk(
        self,
        chunk: str,
        count: int,
        target_language: Optional[str] = None
    ) -> List[QuizQuestion]:
        """Generate `count` questions from one chunk of material via OpenRouter"""
        if target_language:
            translation_rules = f"""
- Additionally translate every question into {target_language}:
  "question_translated", "options_translated" and "explanation_translated"
- "options_translated" must have the SAME number of entries as "options", in the SAME
  ORDER, so that index N means the same answer in both languages
- Translate the MEANING, not word by word. Keep proper nouns and technical terms
  recognisable, adding the original in parentheses where it helps"""
            translation_fields = (
                ', "question_translated": "...", '
                '"options_translated": ["...", "...", "...", "..."], '
                '"explanation_translated": "..."'
            )
        else:
            translation_rules = ''
            translation_fields = ''

        prompt = f"""You are creating a multiple-choice quiz from study material.

MATERIAL:
<<<
{chunk}
>>>

Create exactly {count} questions based ONLY on the material above.

This is a dense quiz: cover the material exhaustively. Every distinct fact, definition,
term, number, name, step, cause and consequence deserves its own question — including
minor details, not just the main points.

Requirements:
- Each question has between {MIN_OPTIONS} and {MAX_OPTIONS} answer options, with EXACTLY ONE correct
- Wrong options must be plausible and on-topic, not obviously absurd
- Every question must test a DIFFERENT fact. Never ask the same thing twice, and never
  reword an earlier question. If you run out of distinct facts, return fewer questions
  rather than repeating yourself
- Questions must be self-contained: never refer to "the text", "the material" or "above"
- "correct_index" is the 0-based index of the correct option
- "explanation" is 1-2 sentences saying why the correct answer is right
- Write questions, options and explanations in the SAME LANGUAGE as the material{translation_rules}

Format as a valid JSON object:
{{
  "questions": [
    {{"question": "...", "options": ["...", "...", "...", "..."], "correct_index": 0, "explanation": "..."{translation_fields}}}
  ]
}}

Respond ONLY with the JSON object, no additional text."""

        content = ''
        try:
            response = await openrouter_instance.client.responses.create(
                model=DEFAULT_MODEL,
                input=[
                    {"role": "user", "content": prompt}
                ],
                reasoning={
                    "effort": "minimal",
                }
            )

            content = response.output_text
            data = json.loads(content)

            questions = []
            for raw in data.get('questions', []):
                question = validate_question(raw)
                if question:
                    questions.append(question)

            logger.info(f"Generated {len(questions)}/{count} questions from a {len(chunk)}-char chunk")
            return questions

        except json.JSONDecodeError as exc:
            logger.error(f"Failed to parse OpenRouter JSON response for quiz chunk: {exc}")
            logger.error(f"Response was: {content}")
            return []
        except Exception as exc:
            logger.error(f"Error generating quiz questions from OpenRouter: {exc}")
            return []

    # ========== Practice Sessions ==========

    async def build_session(self, quiz_id: str, count: int = SESSION_SIZE) -> Optional[List[Dict[str, Any]]]:
        """
        Build a practice session: random questions with shuffled options.
        Returns None if the quiz does not exist.
        """
        quiz = await self.load_quiz(quiz_id)
        if quiz is None:
            return None

        selected = random.sample(quiz.questions, min(count, len(quiz.questions)))

        session = []
        for question in selected:
            # One permutation drives both languages, so index N stays the same answer in each
            order = list(range(len(question.options)))
            random.shuffle(order)

            item = {
                'id': question.id,
                'question': question.question,
                'options': [question.options[i] for i in order],
                'correct_index': order.index(question.correct_index),
                'explanation': question.explanation,
            }

            if question.has_translation:
                item['question_translated'] = question.question_translated
                item['options_translated'] = [question.options_translated[i] for i in order]
                item['explanation_translated'] = question.explanation_translated

            session.append(item)

        logger.info(f"Built session of {len(session)} questions for quiz {quiz_id}, user {self.user_id}")
        return session

    async def record_result(self, quiz_id: str, correct: int, total: int) -> Optional[Dict[str, Any]]:
        """Record the result of a finished session. Returns updated metadata."""
        quiz = await self.load_quiz(quiz_id)
        if quiz is None:
            return None

        quiz.attempts += 1
        quiz.last_score = correct
        quiz.last_total = total
        if quiz.best_score is None or correct > quiz.best_score:
            quiz.best_score = correct

        await self.save_quiz(quiz)
        return quiz.meta_dict()
