"""
Greek Learning Module - Core Business Logic
Manages Greek word learning: storage, OpenAI integration, exercises
"""

import os
import json
import uuid
import asyncio
import random
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum

from logger import logger
from lib.llm import openrouter_instance


class ExerciseDirection(Enum):
    """Direction of translation exercise"""
    GREEK_TO_RUSSIAN = "greek_to_russian"
    RUSSIAN_TO_GREEK = "russian_to_greek"


class ExerciseType(Enum):
    """Type of exercise"""
    SENTENCE_CONTEXT = "sentence_context"  # Find word in sentence context (current type)
    MATCHING_CARDS = "matching_cards"      # Match 10 pairs of words


@dataclass
class Word:
    """Represents a Greek-Russian word pair"""
    id: str
    greek: str
    russian: str
    added_at: str
    level: str = "A2"
    exercise_count: int = 0
    correct_count: int = 0
    word_type: str = ""  # e.g., noun, verb, adjective
    last_practiced: Optional[str] = None

    @staticmethod
    def create(greek: str, russian: str, word_type: str, level: str = "A2") -> "Word":
        """Factory method to create a new word"""
        return Word(
            id=str(uuid.uuid4()),
            greek=greek,
            russian=russian,
            added_at=datetime.utcnow().isoformat() + "Z",
            level=level,
            word_type=word_type
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Word":
        """Create Word from dictionary"""
        return Word(**data)


@dataclass
class LearnedWord:
    """Represents a learned word"""
    id: str
    greek: str
    russian: str
    learned_at: str
    level: str
    total_exercises: int
    total_correct: int
    word_type: str = ""

    @staticmethod
    def from_word(word: Word) -> "LearnedWord":
        """Convert a Word to LearnedWord"""
        return LearnedWord(
            id=word.id,
            greek=word.greek,
            russian=word.russian,
            word_type=word.word_type,
            learned_at=datetime.utcnow().isoformat() + "Z",
            level=word.level,
            total_exercises=word.exercise_count,
            total_correct=word.correct_count
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "LearnedWord":
        """Create LearnedWord from dictionary"""
        return LearnedWord(**data)


@dataclass
class UserStats:
    """User exercise statistics"""
    total_exercises: int = 0
    total_correct: int = 0
    accuracy: float = 0.0
    greek_to_russian: Dict[str, int] = None
    russian_to_greek: Dict[str, int] = None
    last_session: Optional[str] = None
    streak_days: int = 0

    def __post_init__(self):
        if self.greek_to_russian is None:
            self.greek_to_russian = {"total": 0, "correct": 0}
        if self.russian_to_greek is None:
            self.russian_to_greek = {"total": 0, "correct": 0}

    def update_stats(self, direction: ExerciseDirection, is_correct: bool) -> None:
        """Update statistics after an exercise"""
        self.total_exercises += 1
        if is_correct:
            self.total_correct += 1

        if direction == ExerciseDirection.GREEK_TO_RUSSIAN:
            self.greek_to_russian["total"] += 1
            if is_correct:
                self.greek_to_russian["correct"] += 1
        else:
            self.russian_to_greek["total"] += 1
            if is_correct:
                self.russian_to_greek["correct"] += 1

        # Calculate accuracy
        if self.total_exercises > 0:
            self.accuracy = round(self.total_correct / self.total_exercises, 3)

        self.last_session = datetime.utcnow().isoformat() + "Z"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "UserStats":
        """Create UserStats from dictionary"""
        return UserStats(**data)


@dataclass
class Exercise:
    """Represents a generated sentence context exercise"""
    word_id: str
    direction: str
    sentence: str
    sentence_translation: str
    translated_word: str
    correct_answer: str
    internal_marker: str  # The [[marked]] word from OpenAI


@dataclass
class MatchingCardsExercise:
    """Represents a matching cards exercise with word pairs"""
    exercise_type: str = "matching_cards"
    direction: str = ""  # greek_to_russian or russian_to_greek
    pairs: List[Dict[str, str]] = None  # List of {"greek": "word", "russian": "translation", "word_id": "id"}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "exercise_type": self.exercise_type,
            "direction": self.direction,
            "pairs": self.pairs
        }


class GreekLearningManager:
    """
    Manages Greek word learning for a user
    Handles JSON storage, OpenAI integration, exercise generation
    """

    def __init__(self, user_id: int):
        self.user_id = user_id
        self._lock = asyncio.Lock()  # For concurrent file write protection

    # ========== File Path Helpers ==========

    def _get_words_path(self) -> str:
        """Get path to user's current learning words file"""
        filename = os.path.basename(f'{self.user_id}.json')
        return os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', 'greek_words', filename)
        )

    def _get_learned_path(self) -> str:
        """Get path to user's learned words file"""
        filename = os.path.basename(f'{self.user_id}.json')
        return os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', 'greek_learned', filename)
        )

    def _get_stats_path(self) -> str:
        """Get path to user's statistics file"""
        filename = os.path.basename(f'{self.user_id}.json')
        return os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', 'greek_stats', filename)
        )

    # ========== JSON Load/Save Operations ==========

    async def load_words(self) -> List[Word]:
        """Load current learning words from JSON file"""
        path = self._get_words_path()

        if not os.path.exists(path):
            return []

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return [Word.from_dict(w) for w in data.get('words', [])]
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            logger.error(f"Error loading words for user {self.user_id}: {exc}")
            return []

    async def save_words(self, words: List[Word]) -> None:
        """Save current learning words to JSON file"""
        async with self._lock:
            path = self._get_words_path()
            os.makedirs(os.path.dirname(path), exist_ok=True)

            data = {
                "words": [w.to_dict() for w in words],
                "total_count": len(words),
                "last_fetch_at": datetime.utcnow().isoformat() + "Z"
            }

            try:
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            except Exception as exc:
                logger.error(f"Error saving words for user {self.user_id}: {exc}")

    async def load_learned_words(self) -> List[LearnedWord]:
        """Load learned words from JSON file"""
        path = self._get_learned_path()

        if not os.path.exists(path):
            return []

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return [LearnedWord.from_dict(w) for w in data.get('words', [])]
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            logger.error(f"Error loading learned words for user {self.user_id}: {exc}")
            return []

    async def save_learned_words(self, learned_words: List[LearnedWord]) -> None:
        """Save learned words to JSON file"""
        async with self._lock:
            path = self._get_learned_path()
            os.makedirs(os.path.dirname(path), exist_ok=True)

            data = {
                "words": [w.to_dict() for w in learned_words],
                "total_count": len(learned_words)
            }

            try:
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            except Exception as exc:
                logger.error(f"Error saving learned words for user {self.user_id}: {exc}")

    async def load_stats(self) -> UserStats:
        """Load user statistics from JSON file"""
        path = self._get_stats_path()

        if not os.path.exists(path):
            return UserStats()

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return UserStats.from_dict(data)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            logger.error(f"Error loading stats for user {self.user_id}: {exc}")
            return UserStats()

    async def save_stats(self, stats: UserStats) -> None:
        """Save user statistics to JSON file"""
        async with self._lock:
            path = self._get_stats_path()
            os.makedirs(os.path.dirname(path), exist_ok=True)

            try:
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(stats.to_dict(), f, ensure_ascii=False, indent=2)
            except Exception as exc:
                logger.error(f"Error saving stats for user {self.user_id}: {exc}")

    # ========== Word Management Operations ==========

    async def add_words(self, new_words: List[Word]) -> None:
        """Add new words to learning list"""
        current_words = await self.load_words()
        current_words.extend(new_words)
        await self.save_words(current_words)

    async def get_word_by_id(self, word_id: str) -> Optional[Word]:
        """Get a specific word by ID"""
        words = await self.load_words()
        for word in words:
            if word.id == word_id:
                return word
        return None

    async def delete_word(self, word_id: str) -> bool:
        """Delete a word from learning list"""
        words = await self.load_words()
        original_count = len(words)
        words = [w for w in words if w.id != word_id]

        if len(words) < original_count:
            await self.save_words(words)
            return True
        return False

    async def mark_as_learned(self, word_id: str) -> bool:
        """Move word from learning to learned list"""
        words = await self.load_words()
        learned_words = await self.load_learned_words()

        # Find word in learning list
        word_to_learn = None
        remaining_words = []
        for word in words:
            if word.id == word_id:
                word_to_learn = word
            else:
                remaining_words.append(word)

        if not word_to_learn:
            return False

        # Convert to learned word and add to learned list
        learned_word = LearnedWord.from_word(word_to_learn)
        learned_words.append(learned_word)

        # Save both lists
        await self.save_words(remaining_words)
        await self.save_learned_words(learned_words)

        return True

    async def move_to_learning(self, word_id: str) -> bool:
        """Move word from learned back to learning list"""
        learned_words = await self.load_learned_words()
        words = await self.load_words()

        # Find word in learned list
        learned_word = None
        remaining_learned = []
        for lw in learned_words:
            if lw.id == word_id:
                learned_word = lw
            else:
                remaining_learned.append(lw)

        if not learned_word:
            return False

        # Convert back to Word
        word = Word(
            id=learned_word.id,
            greek=learned_word.greek,
            russian=learned_word.russian,
            added_at=datetime.utcnow().isoformat() + "Z",
            level=learned_word.level,
            word_type=learned_word.word_type,
            exercise_count=learned_word.total_exercises,
            correct_count=learned_word.total_correct,
            last_practiced=None
        )
        words.append(word)

        # Save both lists
        await self.save_words(words)
        await self.save_learned_words(remaining_learned)

        return True

    async def update_word_stats(self, word_id: str, is_correct: bool) -> None:
        """Update word statistics after an exercise"""
        words = await self.load_words()

        for word in words:
            if word.id == word_id:
                word.exercise_count += 1
                if is_correct:
                    word.correct_count += 1
                word.last_practiced = datetime.utcnow().isoformat() + "Z"
                break

        await self.save_words(words)

    async def update_user_stats(self, direction: ExerciseDirection, is_correct: bool) -> None:
        """Update user statistics after an exercise"""
        stats = await self.load_stats()
        stats.update_stats(direction, is_correct)
        await self.save_stats(stats)

    async def get_all_learned_word_texts(self) -> List[str]:
        """Get list of all learned Greek words (for exclusion in OpenAI requests)"""
        learned_words = await self.load_learned_words()
        return [w.greek for w in learned_words]

    def _filter_words_by_types(self, words: List[Word], word_types: Optional[List[str]]) -> List[Word]:
        """
        Filter words by word types.

        Args:
            words: List of words to filter
            word_types: List of word types (e.g., ['noun', 'verb']).
                       None or empty list = no filtering (all words)

        Returns:
            Filtered list of words
        """
        if not word_types:
            return words

        # Case-insensitive comparison
        normalized_types = [wt.lower() for wt in word_types]

        return [
            word for word in words
            if word.word_type.lower() in normalized_types
        ]

    async def get_random_word(self, word_types: Optional[List[str]] = None) -> Optional[Word]:
        """
        Get random word from learning list, optionally filtered by type.

        Args:
            word_types: Optional list of word types to filter by

        Returns:
            Random word or None if no words available after filtering
        """
        words = await self.load_words()

        if not words:
            return None

        if word_types:
            words = self._filter_words_by_types(words, word_types)

        if not words:
            return None

        return random.choice(words)

    # ========== OpenRouter/OpenAI Integration ==========

    async def fetch_words_from_openai(self, count: int = 5) -> List[Word]:
        """
        Fetch new Greek-Russian word pairs from OpenAI via OpenRouter
        """
        # Get already learned words to exclude
        learned_greek_words = await self.get_all_learned_word_texts()
        current_words = await self.load_words()
        current_greek_words = [w.greek for w in current_words]

        # Combine for exclusion list
        all_known_words = learned_greek_words + current_greek_words

        # Build exclusion text
        exclusion_text = ""
        if all_known_words:
            exclusion_text = f"\n\nExclude these words that the user already knows:\n{', '.join(all_known_words[:100])}"

        # Build prompt
        prompt = f"""You are a Greek language teacher. Generate {count} Greek words at A2 level (basic everyday vocabulary) with Russian translations.

Requirements:
- Only A2 level words (common, practical everyday vocabulary)
- Accurate Russian translations
- No duplicates from the excluded list{exclusion_text}

Format your response as a valid JSON object with this structure:
{{
  "words": [
    {{"greek": "word_in_greek", "russian": "translation_in_russian", "level": "A2", "type": "verb"}},
    {{"greek": "word_in_greek2", "russian": "translation_in_russian2", "level": "A2", "type": "adjective"}},
    {{"greek": "word_in_greek3", "russian": "translation_in_russian3", "level": "A2", "type": "noun"}}
  ]
}}

Respond ONLY with the JSON object, no additional text."""

        try:
            # Make request to OpenRouter (using GPT-4 mini via OpenRouter)
            response = await openrouter_instance.client.responses.create(
                model="openai/gpt-5-mini",  # Fast and cheap model via OpenRouter
                input=[
                    {"role": "user", "content": prompt}
                ],
                reasoning={
                    "effort": "minimal",
                }
            )

            # Parse response
            content = response.output_text
            logger.info(f"OpenRouter response for fetch_words: {content}")

            # Extract JSON from response
            data = json.loads(content)

            # Convert to Word objects
            new_words = []
            for word_data in data.get('words', []):
                word = Word.create(
                    greek=word_data['greek'],
                    russian=word_data['russian'],
                    level=word_data.get('level', 'A2'),
                    word_type=word_data.get('type', 'unknown')
                )
                new_words.append(word)

            logger.info(f"Fetched {len(new_words)} new Greek words for user {self.user_id}")
            return new_words

        except json.JSONDecodeError as exc:
            logger.error(f"Failed to parse OpenRouter JSON response: {exc}")
            logger.error(f"Response was: {content}")
            return []
        except Exception as exc:
            logger.error(f"Error fetching words from OpenRouter: {exc}")
            return []

    async def generate_matching_cards_exercise(
        self,
        direction: ExerciseDirection,
        word_types: Optional[List[str]] = None
    ) -> Optional[MatchingCardsExercise]:
        """
        Generate matching cards exercise with 10 random word pairs
        Frontend will shuffle and validate the matches

        Args:
            direction: Translation direction
            word_types: Optional filter by word types
        """
        words = await self.load_words()

        # Apply filter if provided
        if word_types:
            words = self._filter_words_by_types(words, word_types)
            logger.info(f"Filtered to {len(words)} words of types: {word_types}")

        if len(words) == 0:
            logger.warning(f"No words available for matching exercise after filtering")
            return None

        # Select 10 words (with repetitions if needed)
        if len(words) < 10:
            logger.info(f"Only {len(words)} words available, using repetitions for matching")
            selected_words = random.choices(words, k=10)
        else:
            selected_words = random.sample(words, 10)

        # Create pairs
        pairs = []
        for word in selected_words:
            pairs.append({
                "word_id": word.id,
                "greek": word.greek,
                "russian": word.russian
            })

        exercise = MatchingCardsExercise(
            exercise_type="matching_cards",
            direction=direction.value,
            pairs=pairs
        )

        logger.info(f"Generated matching cards exercise with {len(pairs)} pairs for user {self.user_id}")
        return exercise

    async def generate_exercise(self, word: Word, direction: ExerciseDirection) -> Exercise:
        """
        Generate exercise sentence using OpenAI via OpenRouter
        Direction: Greek→Russian or Russian→Greek
        """
        if direction == ExerciseDirection.GREEK_TO_RUSSIAN:
            # Generate Greek sentence, ask to find Russian translation
            prompt = f"""Generate a Greek sentence at A2 level using the word "{word.greek}" (meaning in Russian: {word.russian}).

Requirements:
1. Sentence must be at A2 level (simple, clear, everyday language)
2. Sentence must be 20-30 words long to provide good context
3. Mark ONLY the target word "{word.greek}"(in ANY form of the word) with double square brackets: [[{word.greek}]]
4. The marked word can be in any form(plural, case, tense, third person) or a common inflected form
5. Provide Russian translation of the entire sentence
6. Generate correct translation of the marked word into Russian

Format as a valid JSON object:
{{
  "sentence": "Greek sentence with [[target_word]]",
  "translation": "Russian translation of whole sentence with [[russian_word]]"
}}

Respond ONLY with the JSON object, no additional text."""

        else:  # RUSSIAN_TO_GREEK
            # Generate Russian sentence, ask to find Greek translation
            prompt = f"""Generate a Russian sentence at A2 level using the word "{word.russian}" (Greek translation: {word.greek}).

Requirements:
1. Sentence must be simple and clear (A2 level)
2. Sentence must be 20-30 words long to provide good context
3. Mark ONLY the target word "{word.russian}"(in ANY form of the word) with double square brackets: [[{word.russian}]]
4. The marked word can be in any form(plural, case, tense, third person) or a common inflected form
5. Provide Greek translation of the entire sentence
6. In the Greek translation, mark the corresponding Greek word with brackets: [[{word.greek}]]
7. Generate correct Greek translation of the marked word

Format as a valid JSON object:
{{
  "sentence": "Russian sentence with [[target_word]]",
  "translation": "Greek translation with [[greek_word]]"
}}

Respond ONLY with the JSON object, no additional text."""

        try:
            # Make request to OpenRouter
            response = await openrouter_instance.client.responses.create(
                model="openai/gpt-5-mini",
                input=[
                    {"role": "user", "content": prompt}
                ],
                reasoning={
                    "effort": "minimal",
                }
            )

            content = response.output_text
            logger.info(f"OpenRouter response for exercise: {content}")

            # Parse JSON
            data = json.loads(content)

            sentence = data['sentence']
            translation = data['translation']

            # Extract marked word from sentence using regex
            marked_pattern = r'\[\[([^\]]+)\]\]'
            marked_matches = re.findall(marked_pattern, sentence)

            if not marked_matches:
                logger.error("No marked word found in sentence")
                raise ValueError("No marked word in sentence")

            internal_marker = f"[[{marked_matches[0]}]]"
            correct_answer = marked_matches[0]
            translated_word = re.findall(marked_pattern, translation)

            # Extract marked word from translation (for correct answer)
            # translation_matches = re.findall(marked_pattern, translation)
            # if translation_matches:
            #     correct_answer = translation_matches[0]
            # else:
            #     # Fallback to original word
            #     correct_answer = word.russian if direction == ExerciseDirection.GREEK_TO_RUSSIAN else word.greek


            # Remove brackets from sentence for display
            display_sentence = re.sub(marked_pattern, r'\1', sentence)
            display_translation = re.sub(marked_pattern, r'\1', translation)

            exercise = Exercise(
                word_id=word.id,
                direction=direction.value,
                sentence=display_sentence,
                sentence_translation=display_translation,
                translated_word=translated_word,
                correct_answer=correct_answer,
                internal_marker=internal_marker
            )

            logger.info(f"Generated exercise for word {word.greek}/{word.russian}, direction: {direction.value}")
            return exercise

        except json.JSONDecodeError as exc:
            logger.error(f"Failed to parse exercise JSON: {exc}")
            logger.error(f"Response was: {content}")
            raise
        except Exception as exc:
            logger.error(f"Error generating exercise: {exc}")
            raise
