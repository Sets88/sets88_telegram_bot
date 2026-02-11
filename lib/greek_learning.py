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
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
from random import choice

from logger import logger
from lib.llm import openrouter_instance
import replicate_module
from replicate.helpers import FileOutput
import aiohttp


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
    lists: List[str] = None  # List of list IDs this word belongs to

    def __post_init__(self):
        if self.lists is None:
            self.lists = []

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
    lists: List[str] = None  # List of list IDs this word belongs to

    def __post_init__(self):
        if self.lists is None:
            self.lists = []

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
            total_correct=word.correct_count,
            lists=word.lists.copy() if word.lists else []
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


def get_word_hash(greek_word: str) -> str:
    """
    Generate a hash for a Greek word to use as cache filename
    Normalizes the word (lowercase, strip) before hashing
    """
    normalized = greek_word.lower().strip()
    hash_obj = hashlib.md5(normalized.encode('utf-8'))
    return hash_obj.hexdigest()


# Simple in-memory cache for sentence audio URLs with size limit
class SentenceAudioCache:
    """LRU-like cache for sentence audio URLs"""
    def __init__(self, maxsize: int = 100):
        self.maxsize = maxsize
        self.cache = {}  # {text: url}
        self.access_order = []  # Track access order for LRU

    def get(self, text: str) -> Optional[str]:
        """Get cached URL for text"""
        if text in self.cache:
            # Update access order (move to end = most recently used)
            if text in self.access_order:
                self.access_order.remove(text)
            self.access_order.append(text)
            return self.cache[text]
        return None

    def set(self, text: str, url: str) -> None:
        """Cache URL for text"""
        # If already exists, update access order
        if text in self.cache:
            if text in self.access_order:
                self.access_order.remove(text)
        else:
            # If cache is full, remove least recently used
            if len(self.cache) >= self.maxsize:
                if self.access_order:
                    lru_text = self.access_order.pop(0)
                    self.cache.pop(lru_text, None)

        # Add/update cache
        self.cache[text] = url
        self.access_order.append(text)


# Global cache instance
_sentence_audio_cache = SentenceAudioCache(maxsize=100)




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

    def _get_word_forms_cache_path(self, greek_word: str) -> str:
        """Get path to cached word forms file based on word hash"""
        word_hash = get_word_hash(greek_word)
        filename = os.path.basename(f'{word_hash}.json')
        return os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', 'greek_word_forms', filename)
        )

    def _get_audio_cache_dir(self) -> str:
        """Get path to audio cache directory"""
        return os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', 'greek_audio_cache')
        )

    def _get_audio_cache_path(self, text: str, language: str) -> str:
        """Get path to cached audio file based on text hash"""
        # Create hash from text + language
        text_hash = hashlib.md5(f"{text}:{language}".encode('utf-8')).hexdigest()
        filename = os.path.basename(f'{text_hash}.mp3')
        return os.path.join(self._get_audio_cache_dir(), filename)

    async def _download_and_cache_audio(self, audio_url: str, cache_path: str) -> bool:
        """Download audio from URL and save to cache path"""
        try:
            # Create cache directory if it doesn't exist
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)

            # Download audio file
            async with aiohttp.ClientSession() as session:
                async with session.get(audio_url) as response:
                    if response.status != 200:
                        logger.error(f"Failed to download audio: HTTP {response.status}")
                        return False

                    # Save to file
                    with open(cache_path, 'wb') as f:
                        f.write(await response.read())

            logger.info(f"Cached audio file: {cache_path}")
            return True

        except Exception as exc:
            logger.error(f"Error downloading and caching audio: {exc}")
            return False

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

    async def update_word(self, word_id: str, updates: Dict[str, Any]) -> Optional[Word]:
        """Update a word in learning or learned list"""
        # Try learning words first
        words = await self.load_words()
        for word in words:
            if word.id == word_id:
                # Update fields
                for key, value in updates.items():
                    if hasattr(word, key):
                        setattr(word, key, value)
                await self.save_words(words)
                return word

        # Try learned words
        learned_words = await self.load_learned_words()
        for word in learned_words:
            if word.id == word_id:
                # Update fields
                for key, value in updates.items():
                    if hasattr(word, key):
                        setattr(word, key, value)
                await self.save_learned_words(learned_words)
                return word

        return None

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

    async def load_word_forms_cache(self, greek_word: str) -> Optional[List[Dict[str, str]]]:
        """Load cached word forms from file based on word hash"""
        path = self._get_word_forms_cache_path(greek_word)

        if not os.path.exists(path):
            return None

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                word_hash = get_word_hash(greek_word)
                logger.info(f"Loaded word forms from cache for '{greek_word}' (hash: {word_hash})")
                return data.get('forms', [])
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            logger.error(f"Error loading word forms cache for '{greek_word}': {exc}")
            return None

    async def save_word_forms_cache(self, greek: str, russian: str, word_type: str, forms: List[Dict[str, str]]) -> None:
        """Save word forms to cache file based on word hash"""
        async with self._lock:
            path = self._get_word_forms_cache_path(greek)
            os.makedirs(os.path.dirname(path), exist_ok=True)

            word_hash = get_word_hash(greek)
            data = {
                "word_hash": word_hash,
                "greek": greek,
                "russian": russian,
                "word_type": word_type,
                "forms": forms,
                "cached_at": datetime.utcnow().isoformat() + "Z"
            }

            try:
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved word forms to cache for '{greek}' (hash: {word_hash})")
            except Exception as exc:
                logger.error(f"Error saving word forms cache for '{greek}': {exc}")

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
- Modern Greek words (avoid archaic/ancient terms)

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

    async def generate_exercise(
        self,
        word: Word,
        direction: ExerciseDirection,
        verb_preferences: Optional[Dict[str, List[str]]] = None
    ) -> Exercise:
        """
        Generate exercise sentence using OpenAI via OpenRouter
        Direction: Greek→Russian or Russian→Greek
        verb_preferences: Optional dict with 'tenses' and 'persons' lists
        """
        # Build verb form instructions if applicable
        verb_instructions = ""
        if word.word_type.lower() == 'verb' and verb_preferences:
            tenses = verb_preferences.get('tenses', [])
            persons = verb_preferences.get('persons', [])

            if tenses or persons:
                verb_instructions = "\n\nVERB FORM REQUIREMENTS:\n"

                if tenses:
                    tense_map = {
                        'present': 'present tense (ενεστώτας)',
                        'past': 'past tense/aorist (αόριστος)',
                        'future': 'future tense (μέλλοντας)',
                        'other': 'other tenses or moods (e.g., perfect, subjunctive, imperative)'
                    }
                    tense_descriptions = [tense_map.get(t, t) for t in tenses]
                    verb_instructions += f"- Use the verb in ONE of these tenses: {' OR '.join(tense_descriptions)}\n"

                if persons:
                    person_map = {
                        '1st': '1st person (εγώ, εμείς)',
                        '2nd': '2nd person (εσύ, εσείς)',
                        '3rd': '3rd person (αυτός/αυτή/αυτό, αυτοί/αυτές)',
                        'mixed': 'any person (you choose)'
                    }

                    if 'mixed' in persons or len(persons) > 1:
                        person_descriptions = [person_map.get(p, p) for p in persons]
                        verb_instructions += f"- Use ONE of these grammatical persons: {' OR '.join(person_descriptions)}\n"
                    else:
                        # Only one specific person requested
                        person_descriptions = [person_map.get(p, p) for p in persons]
                        verb_instructions += f"- Use this grammatical person: {person_descriptions[0]}\n"

        if direction == ExerciseDirection.GREEK_TO_RUSSIAN:
            # Generate Greek sentence, ask to find Russian translation
            russian_word = choice([x.strip() for x in re.split('[,/]', word.russian) if x.strip()])
            prompt = f"""Generate a Greek sentence at A2 level using the word "{word.greek}" (meaning in Russian: {russian_word}).

Requirements:
1. Sentence must be at A2 level (simple, clear, everyday language)
2. Sentence must be 20-30 words long to provide good context
3. Mark ONLY the target word "{word.greek}"(in ANY form of the word) with double square brackets: [[{word.greek}]]
4. The marked word can be in any form(plural, case, tense, third person) or a common inflected form
5. Provide Russian translation of the entire sentence
6. Generate correct translation of the marked word into Russian{verb_instructions}
7. Modern Greek (avoid archaic/ancient terms)

Format as a valid JSON object:
{{
  "sentence": "Greek sentence with [[target_word]]",
  "translation": "Russian translation of whole sentence with [[russian_word]]"
}}

Respond ONLY with the JSON object, no additional text."""

        else:  # RUSSIAN_TO_GREEK
            # Generate Russian sentence, ask to find Greek translation
            russian_word = choice([x.strip() for x in re.split('[,/]', word.russian) if x.strip()])
            prompt = f"""Generate a Russian sentence at A2 level using the word "{russian_word}" (Greek translation: {word.greek})

Requirements:
1. Sentence must be simple and clear (A2 level)
2. Sentence must be 20-30 words long to provide good context
3. Mark ONLY the target word "{russian_word}"(in ANY form of the word) with double square brackets: [[{russian_word}]]
4. The marked word can be in any form(plural, case, tense, third person) or a common inflected form
5. Provide Greek translation of the entire sentence
6. In the Greek translation, mark the corresponding Greek word with brackets: [[{word.greek}]]
7. Generate correct Greek translation of the marked word{verb_instructions}
8. Modern Greek (avoid archaic/ancient terms)

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

    async def generate_word_forms(self, greek_word: str, russian_word: str, word_type: str) -> List[Dict[str, str]]:
        """
        Generate various forms of a Greek word with Russian translations using OpenAI via OpenRouter
        Returns list of forms with labels
        Uses file cache (based on word hash) to avoid repeated API calls
        """
        # Check cache first (based on word hash, not ID)
        cached_forms = await self.load_word_forms_cache(greek_word)
        if cached_forms is not None:
            logger.info(f"Using cached word forms for '{greek_word}'")
            return cached_forms
        # If word type is unknown, ask OpenAI to determine it and provide forms
        if word_type.lower() == 'unknown' or not word_type:
            prompt = f"""Analyze the Greek word "{greek_word}" and provide its grammatical forms with Russian translations.

IMPORTANT: The word "{greek_word}" may be in ANY form (inflected, conjugated, etc.). You must:
1. Identify the BASE FORM (lemma) of this word
2. Determine what type of word it is (noun, verb, adjective, etc.)
3. Generate all relevant forms based on the word type

Format as a valid JSON array where THE FIRST element is ALWAYS the base form (lemma):
[
  {{"label": "Lemma", "greek": "base_form_here", "russian": "base form translation"}},
  {{"label": "Word type", "greek": "base_form_here", "russian": "type in Russian (существительное/глагол/прилагательное)"}},
  {{"label": "Form description", "greek": "form", "russian": "translation"}},
  ...
]

For example, if input is "έκανα" (past tense), first element should be lemma "κάνω".
Generate all relevant forms:
- For verbs: present, past/aorist, future tenses; 1st, 2nd, 3rd person; singular and plural
- For nouns: nominative, genitive, accusative cases; singular and plural
- For adjectives: masculine, feminine, neuter; singular and plural
- Modern Greek words (avoid archaic/ancient terms)

Respond ONLY with the JSON array, no additional text."""

        # Build prompt based on word type
        elif word_type.lower() == 'verb':
            prompt = f"""Generate all common forms of the Greek verb "{greek_word}" (Russian: {russian_word}) with Russian translations.

IMPORTANT: The word "{greek_word}" may be in ANY form (conjugated). You must:
1. Identify the BASE FORM (lemma/infinitive) - typically 1st person singular present
2. Generate all common forms

Format as a valid JSON array where THE FIRST element is ALWAYS the base form (lemma):
[
  {{"label": "Lemma (1st person singular present)", "greek": "base_form", "russian": "base translation"}},
  {{"label": "Present 1st person singular (εγώ)", "greek": "form", "russian": "translation"}},
  {{"label": "Present 2nd person singular (εσύ)", "greek": "form", "russian": "translation"}},
  ...
]

Include the following forms:
1. Present tense: 1st, 2nd, 3rd person singular and plural
2. Past/Aorist tense: 1st, 2nd, 3rd person singular and plural
3. Future tense: 1st, 2nd, 3rd person singular and plural
4. Imperative mood (if applicable)
5. Participles (if applicable)
6. Modern Greek forms (avoid archaic/ancient terms)

Respond ONLY with the JSON array, no additional text."""

        elif word_type.lower() == 'noun':
            prompt = f"""Generate all forms of the Greek noun "{greek_word}" (Russian: {russian_word}) with Russian translations.

IMPORTANT: The word "{greek_word}" may be in ANY form (case, number). You must:
1. Identify the BASE FORM (lemma) - typically nominative singular
2. Generate all forms

Format as a valid JSON array where THE FIRST element is ALWAYS the base form (lemma):
[
  {{"label": "Lemma (Nominative singular)", "greek": "base_form", "russian": "base translation"}},
  {{"label": "Nominative singular", "greek": "form", "russian": "translation"}},
  {{"label": "Nominative plural", "greek": "form", "russian": "translation"}},
  ...
]

Include the following forms:
1. Nominative singular and plural
2. Genitive singular and plural
3. Accusative singular and plural
4. Vocative (if different)
5. Modern Greek forms (avoid archaic/ancient terms)

Respond ONLY with the JSON array, no additional text."""

        elif word_type.lower() == 'adjective':
            prompt = f"""Generate all forms of the Greek adjective "{greek_word}" (Russian: {russian_word}) with Russian translations.

IMPORTANT: The word "{greek_word}" may be in ANY form (gender, case, number). You must:
1. Identify the BASE FORM (lemma) - typically masculine nominative singular
2. Generate all forms

Format as a valid JSON array where THE FIRST element is ALWAYS the base form (lemma):
[
  {{"label": "Lemma (Masculine nominative singular)", "greek": "base_form", "russian": "base translation"}},
  {{"label": "Masculine singular", "greek": "form", "russian": "translation"}},
  {{"label": "Feminine singular", "greek": "form", "russian": "translation"}},
  ...
]

Include the following forms:
1. Masculine, feminine, neuter forms
2. Singular and plural for each gender
3. Different cases if applicable (nominative, genitive, accusative)
4. Modern Greek forms (avoid archaic/ancient terms)

Respond ONLY with the JSON array, no additional text."""

        else:
            # For other word types, return basic info
            return [{
                "label": "Base form",
                "greek": greek_word,
                "russian": russian_word
            }]

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
            logger.info(f"OpenRouter response for word forms: {content}")

            # Parse JSON
            forms = json.loads(content)

            if not isinstance(forms, list):
                logger.error(f"Expected list, got: {type(forms)}")
                return []

            logger.info(f"Generated {len(forms)} forms for word '{greek_word}'")

            # Save to cache for future use (indexed by word hash)
            await self.save_word_forms_cache(greek_word, russian_word, word_type, forms)

            return forms

        except json.JSONDecodeError as exc:
            logger.error(f"Failed to parse word forms JSON: {exc}")
            logger.error(f"Response was: {content}")
            return []
        except Exception as exc:
            logger.error(f"Error generating word forms: {exc}")
            return []

    async def translate_russian_to_greek(self, russian_word: str) -> Dict[str, str]:
        """
        Translate Russian word to Greek using OpenAI
        Returns dict with 'greek', 'russian', 'word_type'
        """
        prompt = f"""Translate the Russian word "{russian_word}" to Greek.

Provide:
1. The Greek translation (base form)
2. The word type (noun/verb/adjective/etc.)
3. Modern Greek form (avoid archaic/ancient terms)

Format as a valid JSON object:
{{
  "greek": "greek_word",
  "russian": "{russian_word}",
  "word_type": "noun/verb/adjective/etc"
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
            logger.info(f"OpenRouter response for translation: {content}")

            # Parse JSON
            translation = json.loads(content)

            logger.info(f"Translated '{russian_word}' to '{translation.get('greek')}'")
            return translation

        except json.JSONDecodeError as exc:
            logger.error(f"Failed to parse translation JSON: {exc}")
            logger.error(f"Response was: {content}")
            return {}
        except Exception as exc:
            logger.error(f"Error translating Russian to Greek: {exc}")
            return {}

    async def add_custom_word(self, word_text: str, language: str = 'auto') -> Dict[str, Any]:
        """
        Add a custom word entered by user (in Russian or Greek)

        Args:
            word_text: The word to add (in Russian or Greek)
            language: 'russian', 'greek', or 'auto' to detect automatically

        Returns:
            Dict with 'success' and 'word' (Word object as dict) or 'error'
        """
        try:
            word_text = word_text.strip()

            if not word_text:
                return {'success': False, 'error': 'Word cannot be empty'}

            # Auto-detect language if needed
            if language == 'auto':
                # Simple language detection based on character sets
                greek_chars = any('\u0370' <= c <= '\u03FF' or '\u1F00' <= c <= '\u1FFF' for c in word_text)
                russian_chars = any('\u0400' <= c <= '\u04FF' for c in word_text)

                if greek_chars and not russian_chars:
                    language = 'greek'
                elif russian_chars and not greek_chars:
                    language = 'russian'
                else:
                    return {'success': False, 'error': 'Could not detect language. Please enter word in Greek or Russian.'}

            if language == 'russian':
                # Translate Russian to Greek
                translation = await self.translate_russian_to_greek(word_text)

                if not translation or not translation.get('greek'):
                    return {'success': False, 'error': 'Failed to translate word to Greek'}

                greek_word = translation['greek']
                russian_word = word_text
                word_type = translation.get('word_type', 'unknown')

            elif language == 'greek':
                # Get Russian translation and word type from OpenAI
                prompt = f"""Translate the Greek word "{word_text}" to Russian and determine its word type.

Provide:
1. The Russian translation
2. The word type (noun/verb/adjective/etc.)
3. Modern Greek form (avoid archaic/ancient terms)

Format as a valid JSON object:
{{
  "greek": "{word_text}",
  "russian": "russian_translation",
  "word_type": "noun/verb/adjective/etc"
}}

Respond ONLY with the JSON object, no additional text."""

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
                logger.info(f"OpenRouter response for Greek word info: {content}")

                translation = json.loads(content)

                if not translation or not translation.get('russian'):
                    return {'success': False, 'error': 'Failed to translate Greek word to Russian'}

                greek_word = word_text
                russian_word = translation['russian']
                word_type = translation.get('word_type', 'unknown')

            else:
                return {'success': False, 'error': 'Invalid language parameter'}

            # Check if word already exists
            current_words = await self.load_words()
            learned_words = await self.load_learned_words()

            # Check in current learning words
            for w in current_words:
                if w.greek.lower() == greek_word.lower():
                    return {'success': False, 'error': f'Word "{greek_word}" is already in your learning list'}

            # Check in learned words
            for w in learned_words:
                if w.greek.lower() == greek_word.lower():
                    return {'success': False, 'error': f'Word "{greek_word}" is already learned'}

            # Create new word
            new_word = Word.create(
                greek=greek_word,
                russian=russian_word,
                word_type=word_type,
                level="A2"
            )

            # Add to learning list
            await self.add_words([new_word])

            logger.info(f"Added custom word for user {self.user_id}: {greek_word} ({russian_word})")

            return {
                'success': True,
                'word': new_word.to_dict()
            }

        except json.JSONDecodeError as exc:
            logger.error(f"Failed to parse OpenAI JSON response: {exc}")
            return {'success': False, 'error': 'Failed to process word with AI'}
        except Exception as exc:
            logger.error(f"Error adding custom word: {exc}")
            return {'success': False, 'error': f'Internal error: {str(exc)}'}

    def _is_single_word(self, text: str) -> bool:
        """
        Determine if text is a single word or short phrase (≤3 words)
        Single words are cached to disk, sentences use LRU cache for URL
        """
        word_count = len(text.strip().split())
        return word_count <= 3

    async def generate_speech(self, text: str, language: str = 'auto') -> Optional[Dict[str, Any]]:
        """
        Generate speech audio from text using Replicate speech-02-turbo model
        - Single words (≤3 words): cached to disk, returns {'type': 'file', 'path': local_path}
        - Sentences (>3 words): cached URL in LRU memory, returns {'type': 'url', 'path': remote_url}

        Args:
            text: Text to convert to speech
            language: 'greek', 'russian', or 'auto' to detect automatically

        Returns:
            Dict with 'type' ('file' or 'url') and 'path', or None if generation failed
        """
        try:
            # Auto-detect language if needed
            if language == 'auto':
                greek_chars = any('\u0370' <= c <= '\u03FF' or '\u1F00' <= c <= '\u1FFF' for c in text)
                russian_chars = any('\u0400' <= c <= '\u04FF' for c in text)

                if greek_chars and not russian_chars:
                    language = 'greek'
                elif russian_chars and not greek_chars:
                    language = 'russian'
                else:
                    # Default to greek if can't determine
                    language = 'greek'

            # Determine if this is a word or sentence
            is_word = self._is_single_word(text)

            if is_word:
                # Check disk cache for words
                cache_path = self._get_audio_cache_path(text, language)
                if os.path.exists(cache_path):
                    logger.info(f"Audio cache HIT (disk) for word: {text}")
                    return {'type': 'file', 'path': cache_path}
            else:
                # Check LRU cache for sentences
                cached_url = _sentence_audio_cache.get(text)
                if cached_url:
                    logger.info(f"Audio cache HIT (LRU) for sentence: {text[:50]}...")
                    return {'type': 'url', 'path': cached_url}

            # Generate audio using Replicate
            model_info = replicate_module.REPLICATE_MODELS.get('speech-02-turbo')
            if not model_info:
                logger.error("speech-02-turbo model not found in REPLICATE_MODELS")
                return None

            # Prepare input for speech model
            input_data = {
                'text': text,
                'speed': 0.8,
                'emotion': 'neutral'
            }

            logger.info(f"Generating speech for text: {text[:50]}... (language: {language}, is_word: {is_word})")

            # Call replicate_execute synchronously (we'll wrap it in async context)
            result = await asyncio.to_thread(
                replicate_module.replicate_execute,
                model_info['replicate_id'],
                input_data
            )

            # Extract URL from result
            audio_url = None

            if isinstance(result, str):
                # Result is a URL string
                audio_url = result
            elif isinstance(result, FileOutput):
                # Result is FileOutput object with url attribute
                audio_url = result.url
            elif isinstance(result, list) and len(result) > 0:
                # Result is a list, take first item
                first_item = result[0]
                if isinstance(first_item, str):
                    audio_url = first_item
                elif hasattr(first_item, 'url'):
                    audio_url = first_item.url

            if not audio_url:
                logger.error(f"Could not extract audio URL from result: {type(result)}")
                return None

            logger.info(f"Generated speech audio: {audio_url}")

            if is_word:
                # Download and cache to disk for words
                cache_path = self._get_audio_cache_path(text, language)
                success = await self._download_and_cache_audio(audio_url, cache_path)
                if success:
                    return {'type': 'file', 'path': cache_path}
                else:
                    # Failed to cache, return URL anyway
                    return {'type': 'url', 'path': audio_url}
            else:
                # For sentences, cache URL in LRU and return URL
                _sentence_audio_cache.set(text, audio_url)
                return {'type': 'url', 'path': audio_url}

        except Exception as exc:
            logger.error(f"Error generating speech: {exc}")
            return None
