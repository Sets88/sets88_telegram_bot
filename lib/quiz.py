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
MIN_CHUNK_CHARS = 400  # below this a chunk is too thin to be worth its own LLM call
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

def _split_headed_blocks(content: str) -> List[tuple]:
    """
    Split on markdown headings into (block_text, ancestor_heading_lines) pairs.
    The ancestors let a later chunk state which section it came from, so the model
    never sees an orphan fragment with no idea what it is about.
    """
    parts = [b for b in re.split(r'(?m)^(?=#{1,6} )', content) if b.strip()]

    blocks = []
    stack: List[tuple] = []  # [(level, heading_line)]

    for part in parts:
        heading = re.match(r'^(#{1,6})\s+(.*)', part.splitlines()[0])
        ancestors = tuple(h for _, h in stack)
        blocks.append((part, ancestors))

        if heading:
            level = len(heading.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, part.splitlines()[0].strip()))

    return blocks


def split_markdown(content: str) -> List[str]:
    """
    Split markdown into chunks of roughly CHUNK_TARGET_CHARS characters.
    Prefers heading boundaries, then paragraph boundaries, then a raw slice.
    Every chunk is prefixed with the headings of the section it belongs to.
    """
    blocks = _split_headed_blocks(content)
    if not blocks:
        return []

    # Hard-split any block that is bigger than the target on its own
    sized_blocks: List[tuple] = []
    for block, ancestors in blocks:
        if len(block) <= CHUNK_TARGET_CHARS:
            sized_blocks.append((block, ancestors))
            continue

        # Continuations of a split section repeat that section's own heading too
        own = block.splitlines()[0].strip()
        cont_ancestors = ancestors + (own,) if own.startswith('#') else ancestors

        pieces: List[str] = []
        current = ''
        for paragraph in block.split('\n\n'):
            piece = paragraph if not current else f'{current}\n\n{paragraph}'
            if len(piece) <= CHUNK_TARGET_CHARS:
                current = piece
                continue

            if current:
                pieces.append(current)
                current = ''

            # A single paragraph that still does not fit - slice it raw
            while len(paragraph) > CHUNK_TARGET_CHARS:
                pieces.append(paragraph[:CHUNK_TARGET_CHARS])
                paragraph = paragraph[CHUNK_TARGET_CHARS:]
            current = paragraph

        if current:
            pieces.append(current)

        for i, piece in enumerate(pieces):
            sized_blocks.append((piece, ancestors if i == 0 else cont_ancestors))

    # Greedily merge neighbouring blocks up to the target size. A chunk inherits the
    # ancestors of the first block in it - that is the section it opens in.
    chunks: List[tuple] = []
    current, current_ancestors = '', ()
    for block, ancestors in sized_blocks:
        piece = block if not current else f'{current}\n\n{block}'
        # Overshoot the target rather than emit a chunk too thin to quiz on
        if len(piece) <= CHUNK_TARGET_CHARS or len(current) < MIN_CHUNK_CHARS:
            if not current:
                current_ancestors = ancestors
            current = piece
        else:
            chunks.append((current, current_ancestors))
            current, current_ancestors = block, ancestors

    if current:
        # A thin tail belongs with the previous chunk, not on its own
        if chunks and len(current) < MIN_CHUNK_CHARS:
            tail, tail_ancestors = chunks.pop()
            chunks.append((f'{tail}\n\n{current}', tail_ancestors))
        else:
            chunks.append((current, current_ancestors))

    # Prefix each chunk with the headings above it, so it reads as part of a document
    out = []
    for text, ancestors in chunks:
        if not text.strip():
            continue
        missing = [h for h in ancestors if h not in text]
        out.append('\n'.join(missing) + '\n\n' + text if missing else text)

    return out


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


def build_response_schema(translated: bool) -> Dict[str, Any]:
    """
    JSON schema the model is constrained to. This is what actually guarantees a parseable
    response - the model reliably mangles quoting when writing Greek text by hand.
    """
    properties = {
        'question': {'type': 'string'},
        'options': {'type': 'array', 'items': {'type': 'string'}},
        'correct_index': {'type': 'integer'},
        'explanation': {'type': 'string'},
    }

    if translated:
        properties.update({
            'question_translated': {'type': 'string'},
            'options_translated': {'type': 'array', 'items': {'type': 'string'}},
            'explanation_translated': {'type': 'string'},
        })

    return {
        'type': 'object',
        'properties': {
            'questions': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'properties': properties,
                    'required': list(properties),
                    'additionalProperties': False,
                },
            },
        },
        'required': ['questions'],
        'additionalProperties': False,
    }


def salvage_question_objects(content: str) -> List[Dict[str, Any]]:
    """
    Recover whatever question objects can still be parsed from a malformed payload.
    A single stray quote used to cost the entire chunk; this keeps the good ones.
    """
    objects = []
    starts: List[int] = []
    in_string = False
    escaped = False

    for i, ch in enumerate(content):
        if in_string:
            if escaped:
                escaped = False
            elif ch == '\\':
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == '{':
            starts.append(i)
        elif ch == '}' and starts:
            # Objects close innermost-first, so each question is tried before its wrapper
            try:
                obj = json.loads(content[starts.pop():i + 1])
                if isinstance(obj, dict) and 'question' in obj:
                    objects.append(obj)
            except json.JSONDecodeError:
                pass

    return objects


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
            doc_title = quiz.title if quiz else None

            chunks = split_markdown(content)
            counts = plan_question_counts(chunks)
            semaphore = asyncio.Semaphore(GENERATION_CONCURRENCY)

            async def worker(chunk: str, count: int) -> List[QuizQuestion]:
                async with semaphore:
                    return await self._generate_chunk(chunk, count, target_language, doc_title)

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
        target_language: Optional[str] = None,
        doc_title: Optional[str] = None
    ) -> List[QuizQuestion]:
        """Generate `count` questions from one chunk of material via OpenRouter"""
        subject = f'\nSUBJECT OF THE WHOLE DOCUMENT: "{doc_title}"\n' if doc_title else ''
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

        prompt = f"""You are writing multiple-choice quiz questions that will be printed on
flashcards. Each card shows ONE question and its options — nothing else.
{subject}
# THE RULE THAT OVERRIDES EVERYTHING ELSE

The material below is a private reference for YOU ONLY. The person answering has never
seen it, has no access to it, and does not even know a source document exists. As far as
they are concerned, your question is the only text in the world.

So a question may never point at the material. Not "in the text", not "in the excerpt",
not "in the example", not "as mentioned", not "as given", not "according to the passage",
not "στο κείμενο", not "στο απόσπασμα", not "στο παράδειγμα", not "όπως αναφέρεται",
not "όπως δίνεται", not "σύμφωνα με το υλικό", not "в тексте", not "как указано".
Not in ANY language, not in ANY phrasing. If you catch yourself writing such a phrase,
the question is broken: rewrite it so it names its own subject, or delete it.

  BAD:  "Ποια είναι η κατάληξη του πρώτου προσώπου όπως αναφέρεται;"
  GOOD: "Ποια είναι η κατάληξη του πρώτου προσώπου ενικού του ρήματος «γράφω»;"

  BAD:  "Что говорится в тексте о хлоропластах?"
  GOOD: "Какую функцию выполняют хлоропласты в растительной клетке?"

  BAD:  "Что означает эта фраза?"
  GOOD: "Что означает греческая фраза «Είμαι από…»?"

Follow-on rules:
- Name the subject INSIDE the question: the exact term, word, rule or situation asked about.
- Never use bare pointers ("этот", "эта фраза", "this", "these", "the following",
  "αυτό", "το παρακάτω"). Restate the thing in full instead.
- The headings above the excerpt and the document subject tell you the domain. Fold that
  into the wording ONLY when the question would otherwise be ambiguous (e.g. "Στα νέα
  ελληνικά, ..."). Do NOT mechanically prefix every question with the document subject —
  most questions should read naturally without it.
- Ask about the SUBSTANCE, never the presentation: never how many times something is
  repeated, how long a section is, what order items appear in, or how the text is laid out.
- If a fact cannot become a self-contained question, SKIP IT and return fewer questions.
  Fewer good questions beat padding with unanswerable ones.

# MATERIAL (your private reference)
<<<
{chunk}
>>>

Create up to {count} questions based ONLY on the material above.

This is a dense quiz: cover the material exhaustively. Every distinct fact, definition,
term, number, name, step, cause and consequence deserves its own question — including
minor details, not just the main points.

Requirements:
- Each question has between {MIN_OPTIONS} and {MAX_OPTIONS} answer options, with EXACTLY ONE correct
- Wrong options must be plausible and on-topic, not obviously absurd
- Options must also be self-contained: each one readable without the question's context
- Every question must test a DIFFERENT fact. Never ask the same thing twice, and never
  reword an earlier question. If you run out of distinct facts, return fewer questions
  rather than repeating yourself
- "correct_index" is the 0-based index of the correct option
- "explanation" is 1-2 sentences saying why the correct answer is right
- Write questions, options and explanations in the SAME LANGUAGE as the material{translation_rules}

FINAL CHECK before you answer. Re-read every question you wrote and ask:
1. Does it contain any phrase pointing at a text, excerpt, example, passage or "as
   mentioned/given/stated"? If yes — rewrite it without that phrase, or delete it.
2. Could someone who has never seen the material answer it from the question alone?
   If no — rewrite it so it names its own subject, or delete it.
Deleting is always allowed. Returning fewer questions is the correct outcome.

Format as a valid JSON object:
{{
  "questions": [
    {{"question": "...", "options": ["...", "...", "...", "..."], "correct_index": 0, "explanation": "..."{translation_fields}}}
  ]
}}

Respond ONLY with the JSON object, no additional text."""

        schema = build_response_schema(bool(target_language))
        best: List[QuizQuestion] = []

        # The schema should make a malformed payload impossible, but a truncated or
        # otherwise broken response still costs the whole chunk - so salvage and retry once.
        for attempt in range(2):
            questions, clean = await self._request_chunk(prompt, schema)

            if len(questions) > len(best):
                best = questions

            if clean or len(best) >= count:
                break

            if attempt == 0:
                logger.warning(
                    f"Retrying chunk after a malformed response "
                    f"(salvaged {len(best)}/{count} questions)"
                )

        logger.info(f"Generated {len(best)}/{count} questions from a {len(chunk)}-char chunk")
        return best

    async def _request_chunk(self, prompt: str, schema: Dict[str, Any]) -> tuple:
        """One LLM call. Returns (questions, response_was_valid_json)."""
        content = ''
        clean = True

        try:
            response = await openrouter_instance.client.responses.create(
                model=DEFAULT_MODEL,
                input=[
                    {"role": "user", "content": prompt}
                ],
                reasoning={
                    "effort": "minimal",
                },
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "quiz_questions",
                        "strict": True,
                        "schema": schema,
                    }
                }
            )

            content = response.output_text

            try:
                raws = json.loads(content).get('questions', [])
            except json.JSONDecodeError as exc:
                clean = False
                raws = salvage_question_objects(content)
                logger.error(
                    f"Malformed JSON from OpenRouter ({exc}); salvaged {len(raws)} question objects"
                )
                if not raws:
                    logger.error(f"Response was: {content}")

            questions = []
            for raw in raws:
                question = validate_question(raw)
                if question:
                    questions.append(question)

            return questions, clean

        except Exception as exc:
            logger.error(f"Error generating quiz questions from OpenRouter: {exc}")
            return [], False

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
