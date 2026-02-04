// Greek Learning Web App - Main Application Logic

// Initialize Telegram WebApp SDK
const tg = window.Telegram?.WebApp;

if (!tg) {
    console.error('Telegram WebApp SDK not found!');
    alert('This app must be opened from Telegram');
} else {
    console.log('Telegram WebApp initialized');
    console.log('initData:', tg.initData ? 'present' : 'missing');
    tg.expand(); // Expand to full screen
    tg.ready();
}

// App State
const state = {
    learningWords: [],
    learnedWords: [],
    stats: {},
    currentExercise: null,
    exerciseCount: 0,
    exerciseType: null,
    selectedWordType: '', // Word type filter: '' = all, 'noun', 'verb', 'adjective'
    verbFormPreferences: {
        tenses: ['present'],
        persons: ['1st']
    },
    matchingState: {
        selectedLeft: null,
        selectedRight: null,
        matchedPairs: [],
        incorrectAttempts: 0
    },
    searchQueries: {
        learning: '',
        learned: '',
        listWords: ''
    },
    lists: [],  // All unique list names
    currentList: null,  // Currently selected list for management
    selectedListForPractice: null  // List selected for practice
};

// ========== API Helper Functions ==========

async function apiRequest(endpoint, options = {}) {
    const url = `/greek/api/${endpoint}`;

    console.log(`API Request: ${options.method || 'GET'} ${url}`);
    console.log('initData:', tg?.initData ? 'present' : 'MISSING!');

    try {
        const response = await fetch(url, {
            ...options,
            headers: {
                'Content-Type': 'application/json',
                'X-Telegram-Init-Data': tg?.initData || '',
                ...options.headers
            }
        });

        console.log(`API Response: ${response.status} ${response.statusText}`);

        if (!response.ok) {
            let errorMsg = 'API request failed';
            try {
                const error = await response.json();
                errorMsg = error.error || errorMsg;
            } catch (e) {
                errorMsg = `${response.status} ${response.statusText}`;
            }
            throw new Error(errorMsg);
        }

        const data = await response.json();
        console.log('API Response data:', data);
        return data;

    } catch (error) {
        console.error('API Request failed:', error);
        throw error;
    }
}

async function loadWords() {
    try {
        console.log('Loading words...');
        const data = await apiRequest('words');
        state.learningWords = data.words || [];
        console.log(`Loaded ${state.learningWords.length} words`);
        renderLearningWords();
    } catch (error) {
        console.error('Error loading words:', error);
        state.learningWords = [];
        renderLearningWords();
        throw error; // Re-throw to be caught by initApp
    }
}

async function loadLearnedWords() {
    try {
        console.log('Loading learned words...');
        const data = await apiRequest('learned');
        state.learnedWords = data.words || [];
        console.log(`Loaded ${state.learnedWords.length} learned words`);
        renderLearnedWords();
    } catch (error) {
        console.error('Error loading learned words:', error);
        state.learnedWords = [];
        renderLearnedWords();
        throw error; // Re-throw to be caught by initApp
    }
}

async function loadStats() {
    try {
        console.log('Loading stats...');
        const data = await apiRequest('stats');
        state.stats = data;
        console.log('Stats loaded:', state.stats);
        renderStats();
    } catch (error) {
        console.error('Error loading stats:', error);
        state.stats = {
            total_exercises: 0,
            total_correct: 0,
            accuracy: 0,
            greek_to_russian: {total: 0, correct: 0},
            russian_to_greek: {total: 0, correct: 0},
            streak_days: 0
        };
        renderStats();
        throw error; // Re-throw to be caught by initApp
    }
}

async function addCustomWord(wordText) {
    try {
        const loadingEl = document.getElementById('add-word-loading');
        const errorEl = document.getElementById('add-word-error');

        // Show loading, hide error
        loadingEl.classList.remove('hidden');
        errorEl.classList.add('hidden');

        const data = await apiRequest('add-word', {
            method: 'POST',
            body: JSON.stringify({
                word: wordText,
                language: 'auto'
            })
        });

        // Hide loading
        loadingEl.classList.add('hidden');

        if (data.success) {
            // Reload words list
            await loadWords();
            hideAddCustomWordModal();
            tg.showAlert(`Word added successfully: ${data.word.greek} - ${data.word.russian}`);
        } else {
            // Show error
            errorEl.textContent = data.error || 'Failed to add word';
            errorEl.classList.remove('hidden');
        }
    } catch (error) {
        console.error('Error adding custom word:', error);
        const loadingEl = document.getElementById('add-word-loading');
        const errorEl = document.getElementById('add-word-error');

        loadingEl.classList.add('hidden');
        errorEl.textContent = error.message || 'Failed to add word';
        errorEl.classList.remove('hidden');
    }
}

async function fetchNewWords(count) {
    try {
        showLoading();
        const data = await apiRequest('fetch-words', {
            method: 'POST',
            body: JSON.stringify({ count })
        });

        state.learningWords = data.words || state.learningWords;
        await loadWords();
        hideLoading();
        tg.showAlert(`Successfully fetched ${data.new_words?.length || count} new words!`);
    } catch (error) {
        console.error('Error fetching words:', error);
        hideLoading();
        tg.showAlert('Failed to fetch new words');
    }
}

function updateSelectedWordType() {
    const select = document.getElementById('word-type-select');
    if (select) {
        state.selectedWordType = select.value;
        console.log('Selected word type:', state.selectedWordType || 'all');

        // Show/hide verb form section
        const verbFormSection = document.getElementById('verb-form-section');
        if (verbFormSection) {
            if (select.value === 'verb') {
                verbFormSection.classList.remove('hidden');
            } else {
                verbFormSection.classList.add('hidden');
            }
        }
    }
}

function getVerbFormPreferences() {
    const tenseCheckboxes = document.querySelectorAll('input[name="tense"]:checked');
    const personCheckboxes = document.querySelectorAll('input[name="person"]:checked');

    const tenses = Array.from(tenseCheckboxes).map(cb => cb.value);
    const persons = Array.from(personCheckboxes).map(cb => cb.value);

    // Default to all if none selected
    return {
        tenses: tenses.length > 0 ? tenses : ['present', 'past', 'future', 'other'],
        persons: persons.length > 0 ? persons : ['1st', '2nd', '3rd', 'mixed']
    };
}

async function deleteWord(wordId) {
    try {
        await apiRequest(`words/${wordId}`, { method: 'DELETE' });
        await loadWords();
        tg.showAlert('Word deleted');
    } catch (error) {
        console.error('Error deleting word:', error);
        tg.showAlert('Failed to delete word');
    }
}

async function markWordAsLearned(wordId) {
    try {
        await apiRequest('mark-learned', {
            method: 'POST',
            body: JSON.stringify({ word_id: wordId })
        });
        await loadWords();
        await loadLearnedWords();
        tg.showAlert('Word marked as learned! 🎉');
    } catch (error) {
        console.error('Error marking as learned:', error);
        tg.showAlert('Failed to mark word as learned');
    }
}

async function moveToLearning(wordId) {
    try {
        await apiRequest('move-to-learning', {
            method: 'POST',
            body: JSON.stringify({ word_id: wordId })
        });
        await loadWords();
        await loadLearnedWords();
        tg.showAlert('Word moved back to learning');
    } catch (error) {
        console.error('Error moving to learning:', error);
        tg.showAlert('Failed to move word');
    }
}

async function editWord(wordId, updates) {
    try {
        const data = await apiRequest(`words/${wordId}`, {
            method: 'PATCH',
            body: JSON.stringify(updates)
        });
        return data;
    } catch (error) {
        console.error('Error editing word:', error);
        throw error;
    }
}

async function getExercise(wordId = null, direction = 'random') {
    try {
        // Show loading spinner
        showExerciseLoading();

        // If practicing a list, select a random word from the list
        if (state.selectedListForPractice && !wordId) {
            const words = getWordsByList(state.selectedListForPractice);
            const listWords = words.learning;  // Only use learning words for practice

            if (listWords.length === 0) {
                throw new Error('No learning words in this list');
            }

            const randomWord = listWords[Math.floor(Math.random() * listWords.length)];
            wordId = randomWord.id;
        }

        let url = 'exercise?';
        if (wordId) url += `word_id=${wordId}&`;
        url += `direction=${direction}`;

        // Add word type filter if selected
        if (state.selectedWordType) {
            url += `&word_type=${state.selectedWordType}`;

            // Add verb form preferences if word type is verb
            if (state.selectedWordType === 'verb') {
                const prefs = getVerbFormPreferences();
                state.verbFormPreferences = prefs;  // Store in state

                url += `&verb_tenses=${prefs.tenses.join(',')}`;
                url += `&verb_persons=${prefs.persons.join(',')}`;
            }
        }

        const data = await apiRequest(url);
        state.currentExercise = data;
        state.exerciseCount++;

        // Hide loading spinner and render exercise
        hideExerciseLoading();
        renderExercise();
    } catch (error) {
        console.error('Error getting exercise:', error);
        hideExerciseLoading();
        tg.showAlert('Failed to generate exercise');
        showMainScreen();
    }
}

async function validateAnswer(userAnswer) {
    try {
        const data = await apiRequest('validate', {
            method: 'POST',
            body: JSON.stringify({
                word_id: state.currentExercise.word_id,
                user_answer: userAnswer,
                correct_answer: state.currentExercise.correct_answer,
                direction: state.currentExercise.direction
            })
        });

        return data;
    } catch (error) {
        console.error('Error validating answer:', error);
        throw error;
    }
}

async function getMatchingExercise(direction = 'greek_to_russian') {
    try {
        showExerciseLoading();

        let url = `exercise/matching?direction=${direction}`;

        // Add word type filter if selected
        if (state.selectedWordType) {
            url += `&word_type=${state.selectedWordType}`;
        }

        const data = await apiRequest(url);

        state.currentExercise = data;
        state.exerciseType = 'matching_cards';
        state.exerciseCount++;

        // Reset matching state
        state.matchingState = {
            selectedLeft: null,
            selectedRight: null,
            matchedPairs: [],
            incorrectAttempts: 0
        };

        hideExerciseLoading();
        renderMatchingExercise();
    } catch (error) {
        console.error('Error getting matching exercise:', error);
        hideExerciseLoading();
        tg.showAlert('Failed to generate matching exercise');
        showMainScreen();
    }
}

async function validateMatchingResults(results) {
    try {
        await apiRequest('validate-matching', {
            method: 'POST',
            body: JSON.stringify({
                results: results,
                direction: state.currentExercise.direction
            })
        });
    } catch (error) {
        console.error('Error validating matching results:', error);
    }
}

async function updateWordLists(wordId, lists) {
    try {
        const data = await apiRequest(`words/${wordId}`, {
            method: 'PATCH',
            body: JSON.stringify({ lists })
        });
        return data;
    } catch (error) {
        console.error('Error updating word lists:', error);
        throw error;
    }
}

// ========== Lists Management ==========

function getAllListNames() {
    /**
     * Get all unique list names from all words
     */
    const listsSet = new Set();

    state.learningWords.forEach(word => {
        if (word.lists && Array.isArray(word.lists)) {
            word.lists.forEach(listName => listsSet.add(listName));
        }
    });

    state.learnedWords.forEach(word => {
        if (word.lists && Array.isArray(word.lists)) {
            word.lists.forEach(listName => listsSet.add(listName));
        }
    });

    return Array.from(listsSet).sort();
}

function getListStats(listName) {
    /**
     * Get statistics for a list
     */
    let learningCount = 0;
    let learnedCount = 0;

    state.learningWords.forEach(word => {
        if (word.lists && word.lists.includes(listName)) {
            learningCount++;
        }
    });

    state.learnedWords.forEach(word => {
        if (word.lists && word.lists.includes(listName)) {
            learnedCount++;
        }
    });

    return {
        learning: learningCount,
        learned: learnedCount,
        total: learningCount + learnedCount
    };
}

function getWordsByList(listName) {
    /**
     * Get all words (learning + learned) for a specific list
     */
    const learning = state.learningWords.filter(w =>
        w.lists && w.lists.includes(listName)
    );
    const learned = state.learnedWords.filter(w =>
        w.lists && w.lists.includes(listName)
    );

    return { learning, learned };
}

async function addWordToList(wordId, listName) {
    /**
     * Add a word to a list
     */
    // Find word in learning or learned
    let word = state.learningWords.find(w => w.id === wordId);
    let isLearning = true;

    if (!word) {
        word = state.learnedWords.find(w => w.id === wordId);
        isLearning = false;
    }

    if (!word) {
        console.error('Word not found:', wordId);
        return false;
    }

    // Initialize lists if needed
    if (!word.lists) {
        word.lists = [];
    }

    // Add list if not already present
    if (!word.lists.includes(listName)) {
        word.lists.push(listName);

        // Update on backend
        try {
            await updateWordLists(wordId, word.lists);

            // Refresh lists
            state.lists = getAllListNames();
            renderLists();

            return true;
        } catch (error) {
            // Revert on error
            word.lists = word.lists.filter(l => l !== listName);
            console.error('Failed to add word to list:', error);
            return false;
        }
    }

    return true;
}

async function removeWordFromList(wordId, listName) {
    /**
     * Remove a word from a list
     */
    // Find word in learning or learned
    let word = state.learningWords.find(w => w.id === wordId);

    if (!word) {
        word = state.learnedWords.find(w => w.id === wordId);
    }

    if (!word || !word.lists) {
        return false;
    }

    // Remove list
    const index = word.lists.indexOf(listName);
    if (index > -1) {
        word.lists.splice(index, 1);

        // Update on backend
        try {
            await updateWordLists(wordId, word.lists);

            // Refresh lists
            state.lists = getAllListNames();
            renderLists();

            return true;
        } catch (error) {
            // Revert on error
            word.lists.push(listName);
            console.error('Failed to remove word from list:', error);
            return false;
        }
    }

    return false;
}

async function renameList(oldName, newName) {
    /**
     * Rename a list across all words
     */
    if (!oldName || !newName || oldName === newName) {
        return false;
    }

    const updates = [];

    // Update all words with this list
    const allWords = [...state.learningWords, ...state.learnedWords];

    for (const word of allWords) {
        if (word.lists && word.lists.includes(oldName)) {
            const index = word.lists.indexOf(oldName);
            word.lists[index] = newName;
            updates.push(updateWordLists(word.id, word.lists));
        }
    }

    try {
        await Promise.all(updates);

        // Refresh lists
        state.lists = getAllListNames();
        renderLists();

        return true;
    } catch (error) {
        console.error('Failed to rename list:', error);
        // Reload words to revert
        await loadWords();
        await loadLearnedWords();
        return false;
    }
}

async function deleteList(listName) {
    /**
     * Delete a list from all words
     */
    const updates = [];

    // Remove list from all words
    const allWords = [...state.learningWords, ...state.learnedWords];

    for (const word of allWords) {
        if (word.lists && word.lists.includes(listName)) {
            word.lists = word.lists.filter(l => l !== listName);
            updates.push(updateWordLists(word.id, word.lists));
        }
    }

    try {
        await Promise.all(updates);

        // Refresh lists
        state.lists = getAllListNames();
        renderLists();

        return true;
    } catch (error) {
        console.error('Failed to delete list:', error);
        // Reload words to revert
        await loadWords();
        await loadLearnedWords();
        return false;
    }
}

// ========== Word Details & Long Press ==========

function detectLanguage(text) {
    /**
     * Detect if text is Greek or Russian
     * Returns: 'greek', 'russian', or 'unknown'
     */
    const greekRegex = /[\u0370-\u03FF\u1F00-\u1FFF]/; // Greek characters
    const russianRegex = /[\u0400-\u04FF]/; // Cyrillic characters

    const hasGreek = greekRegex.test(text);
    const hasRussian = russianRegex.test(text);

    if (hasGreek && !hasRussian) {
        return 'greek';
    } else if (hasRussian && !hasGreek) {
        return 'russian';
    } else if (hasGreek && hasRussian) {
        // Mixed - count which has more characters
        const greekCount = (text.match(greekRegex) || []).length;
        const russianCount = (text.match(russianRegex) || []).length;
        return greekCount > russianCount ? 'greek' : 'russian';
    } else {
        return 'unknown';
    }
}

async function getWordDetails(wordId) {
    try {
        const data = await apiRequest(`word-details/${wordId}`);
        return data;
    } catch (error) {
        console.error('Error getting word details:', error);
        throw error;
    }
}

async function getWordFormsByText(greekWord, russianWord = '', wordType = 'unknown') {
    try {
        const data = await apiRequest('word-forms', {
            method: 'POST',
            body: JSON.stringify({
                greek: greekWord,
                russian: russianWord,
                word_type: wordType
            })
        });
        return data;
    } catch (error) {
        console.error('Error getting word forms by text:', error);
        throw error;
    }
}

async function translateRussianToGreek(russianWord) {
    try {
        const data = await apiRequest('translate-russian', {
            method: 'POST',
            body: JSON.stringify({
                russian: russianWord
            })
        });
        return data;
    } catch (error) {
        console.error('Error translating Russian to Greek:', error);
        throw error;
    }
}

async function speakText(text, language = 'auto') {
    try {
        const url = `/greek/api/speak`;

        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-Telegram-Init-Data': tg?.initData || ''
            },
            body: JSON.stringify({
                text: text,
                language: language
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error ${response.status}`);
        }

        // Check content type to determine response type
        const contentType = response.headers.get('Content-Type');

        if (contentType && contentType.includes('audio/mpeg')) {
            // Response is audio file (blob) - for words
            const blob = await response.blob();
            return { type: 'blob', data: blob };
        } else {
            // Response is JSON with URL - for sentences
            const json = await response.json();
            return { type: 'url', data: json.audio_url };
        }
    } catch (error) {
        console.error('Error generating speech:', error);
        throw error;
    }
}

async function playSpeech(text, language = 'auto', buttonElement = null) {
    try {
        // Disable button and show loading state
        if (buttonElement) {
            buttonElement.disabled = true;
            buttonElement.classList.add('loading');
        }

        // Request speech audio
        const result = await speakText(text, language);

        let audioUrl;
        let needsCleanup = false;

        if (result.type === 'blob') {
            // Create object URL from blob (words)
            audioUrl = URL.createObjectURL(result.data);
            needsCleanup = true;
        } else if (result.type === 'url') {
            // Use remote URL directly (sentences)
            audioUrl = result.data;
            needsCleanup = false;
        } else {
            throw new Error('Unknown audio result type');
        }

        // Create and play audio
        const audio = new Audio(audioUrl);

        audio.onended = () => {
            // Re-enable button when audio finishes
            if (buttonElement) {
                buttonElement.disabled = false;
                buttonElement.classList.remove('loading');
            }
            // Clean up object URL if needed
            if (needsCleanup) {
                URL.revokeObjectURL(audioUrl);
            }
        };

        audio.onerror = (error) => {
            console.error('Error playing audio:', error);
            if (buttonElement) {
                buttonElement.disabled = false;
                buttonElement.classList.remove('loading');
            }
            // Clean up object URL if needed
            if (needsCleanup) {
                URL.revokeObjectURL(audioUrl);
            }
            tg.showAlert('Failed to play audio');
        };

        await audio.play();
    } catch (error) {
        console.error('Error playing speech:', error);
        if (buttonElement) {
            buttonElement.disabled = false;
            buttonElement.classList.remove('loading');
        }
        tg.showAlert('Failed to generate speech');
    }
}

function attachLongPressHandlers() {
    const wordCards = document.querySelectorAll('.word-card');

    wordCards.forEach(card => {
        let pressTimer = null;
        let isLongPress = false;

        const startPress = (e) => {
            // Don't trigger on button clicks
            if (e.target.closest('.word-actions')) {
                return;
            }

            isLongPress = false;
            pressTimer = setTimeout(() => {
                isLongPress = true;
                const wordId = card.dataset.wordId;
                if (wordId) {
                    showWordDetailsModal(wordId);
                    // Haptic feedback
                    if (tg.HapticFeedback) {
                        tg.HapticFeedback.impactOccurred('medium');
                    }
                }
            }, 500); // 500ms for long press
        };

        const cancelPress = () => {
            if (pressTimer) {
                clearTimeout(pressTimer);
                pressTimer = null;
            }
        };

        // Touch events
        card.addEventListener('touchstart', startPress, { passive: true });
        card.addEventListener('touchend', cancelPress);
        card.addEventListener('touchmove', cancelPress);
        card.addEventListener('touchcancel', cancelPress);

        // Mouse events (for desktop testing)
        card.addEventListener('mousedown', startPress);
        card.addEventListener('mouseup', cancelPress);
        card.addEventListener('mouseleave', cancelPress);
    });
}

function attachLongPressToSentenceWords() {
    const sentenceWords = document.querySelectorAll('.sentence-word');

    sentenceWords.forEach(wordEl => {
        let pressTimer = null;

        const startPress = (e) => {
            // Prevent triggering click during long press
            e.stopPropagation();

            pressTimer = setTimeout(() => {
                const wordId = wordEl.dataset.wordId;
                const wordText = wordEl.dataset.word;

                if (wordId) {
                    // We have word_id, show details directly
                    showWordDetailsModal(wordId);
                } else if (wordText) {
                    // Try to find word by text in current exercise
                    showWordDetailsModalByText(wordText);
                }

                // Haptic feedback
                if (tg.HapticFeedback) {
                    tg.HapticFeedback.impactOccurred('medium');
                }
            }, 500); // 500ms for long press
        };

        const cancelPress = () => {
            if (pressTimer) {
                clearTimeout(pressTimer);
                pressTimer = null;
            }
        };

        // Touch events
        wordEl.addEventListener('touchstart', startPress, { passive: false });
        wordEl.addEventListener('touchend', cancelPress);
        wordEl.addEventListener('touchmove', cancelPress);
        wordEl.addEventListener('touchcancel', cancelPress);

        // Mouse events (for desktop testing)
        wordEl.addEventListener('mousedown', startPress);
        wordEl.addEventListener('mouseup', cancelPress);
        wordEl.addEventListener('mouseleave', cancelPress);
    });
}

function attachLongPressToMatchingCards() {
    const matchingCards = document.querySelectorAll('.matching-card');

    matchingCards.forEach(card => {
        let pressTimer = null;

        const startPress = (e) => {
            e.stopPropagation();

            // Don't trigger on already matched cards
            if (card.classList.contains('matched')) {
                return;
            }

            pressTimer = setTimeout(() => {
                const wordId = card.dataset.id;
                if (wordId) {
                    showWordDetailsModal(wordId);

                    // Haptic feedback
                    if (tg.HapticFeedback) {
                        tg.HapticFeedback.impactOccurred('medium');
                    }
                }
            }, 500);
        };

        const cancelPress = () => {
            if (pressTimer) {
                clearTimeout(pressTimer);
                pressTimer = null;
            }
        };

        // Touch events
        card.addEventListener('touchstart', startPress, { passive: false });
        card.addEventListener('touchend', cancelPress);
        card.addEventListener('touchmove', cancelPress);
        card.addEventListener('touchcancel', cancelPress);

        // Mouse events
        card.addEventListener('mousedown', startPress);
        card.addEventListener('mouseup', cancelPress);
        card.addEventListener('mouseleave', cancelPress);
    });
}

async function showWordDetailsModal(wordId) {
    const modal = document.getElementById('word-details-modal');
    const loading = document.getElementById('word-details-loading');
    const content = document.getElementById('word-details-content');

    // Find word in state
    let word = state.learningWords.find(w => w.id === wordId);
    if (!word) {
        word = state.learnedWords.find(w => w.id === wordId);
    }

    if (!word) {
        console.error('Word not found:', wordId);
        return;
    }

    // Show modal
    modal.classList.remove('hidden');

    // Set basic info
    document.getElementById('word-details-title').textContent = 'Word Details';
    document.getElementById('word-details-greek').textContent = word.greek;
    document.getElementById('word-details-russian').textContent = word.russian;
    document.getElementById('word-details-type').textContent = word.word_type || 'unknown';

    // Show loading
    loading.classList.remove('hidden');
    content.style.opacity = '0.5';

    try {
        // Fetch word forms from API
        const details = await getWordDetails(wordId);

        // Hide loading
        loading.classList.add('hidden');
        content.style.opacity = '1';

        // Display word forms
        if (details.forms && details.forms.length > 0) {
            const formsSection = document.getElementById('word-forms-section');
            const formsList = document.getElementById('word-forms-list');

            formsSection.classList.remove('hidden');

            formsList.innerHTML = details.forms.map(form => `
                <div class="word-form-item">
                    <div class="word-form-label">${form.label}:</div>
                    <div class="word-form-value">
                        <span class="word-form-greek">${form.greek}</span>
                        <span class="word-form-russian">${form.russian}</span>
                    </div>
                </div>
            `).join('');
        } else {
            document.getElementById('word-forms-section').classList.add('hidden');
        }
    } catch (error) {
        console.error('Error loading word details:', error);
        loading.classList.add('hidden');
        content.style.opacity = '1';
        tg.showAlert('Failed to load word details');
    }
}

async function showWordDetailsModalByText(wordText) {
    // Detect language first
    const language = detectLanguage(wordText);
    console.log(`Detected language for "${wordText}": ${language}`);

    // Try to find word in learning or learned lists by matching text
    const normalizeText = (text) => text.toLowerCase().trim();
    const normalizedSearch = normalizeText(wordText);

    // Search in learning words
    let word = state.learningWords.find(w =>
        normalizeText(w.greek).includes(normalizedSearch) ||
        normalizeText(w.russian).includes(normalizedSearch)
    );

    // If not found, search in learned words
    if (!word) {
        word = state.learnedWords.find(w =>
            normalizeText(w.greek).includes(normalizedSearch) ||
            normalizeText(w.russian).includes(normalizedSearch)
        );
    }

    if (word) {
        // Found word in user's lists, show details using word_id
        await showWordDetailsModal(word.id);
    } else if (language === 'russian') {
        // Russian word not in user's list - translate to Greek first
        console.log(`Russian word "${wordText}" not found in lists, translating to Greek first`);
        await showWordDetailsModalForRussianWord(wordText);
    } else if (language === 'greek') {
        // Greek word not in user's list - get forms directly
        console.log(`Greek word "${wordText}" not found in lists, requesting forms for arbitrary word`);
        await showWordDetailsModalForArbitraryWord(wordText);
    } else {
        // Unknown language
        if (tg.showAlert) {
            tg.showAlert('Could not detect word language');
        }
    }
}

async function showWordDetailsModalForRussianWord(russianWord) {
    const modal = document.getElementById('word-details-modal');
    const loading = document.getElementById('word-details-loading');
    const content = document.getElementById('word-details-content');

    // Show modal
    modal.classList.remove('hidden');

    // Set basic info
    document.getElementById('word-details-title').innerHTML = '🔍 Translating <span style="font-size: 12px; font-weight: normal; color: #999;">(Russian word)</span>';
    document.getElementById('word-details-greek').textContent = '(translating...)';
    document.getElementById('word-details-russian').textContent = russianWord;
    document.getElementById('word-details-type').textContent = 'translating...';

    // Show loading
    loading.classList.remove('hidden');
    content.style.opacity = '0.5';

    try {
        // First, translate Russian to Greek
        const translation = await translateRussianToGreek(russianWord);

        if (!translation.greek) {
            throw new Error('Translation failed');
        }

        // Update with Greek translation
        document.getElementById('word-details-greek').textContent = translation.greek;
        document.getElementById('word-details-type').textContent = translation.word_type || 'unknown';
        document.getElementById('word-details-title').innerHTML = '🔍 Word Forms <span style="font-size: 12px; font-weight: normal; color: #999;">(translated from Russian)</span>';

        // Now fetch word forms for the Greek word
        const details = await getWordFormsByText(translation.greek, russianWord, translation.word_type);

        // Hide loading
        loading.classList.add('hidden');
        content.style.opacity = '1';

        // Display word forms
        if (details.forms && details.forms.length > 0) {
            const formsSection = document.getElementById('word-forms-section');
            const formsList = document.getElementById('word-forms-list');

            formsSection.classList.remove('hidden');

            // Skip first item if it's just the word type info
            const formsToDisplay = details.forms[0] && details.forms[0].label === 'Word type'
                ? details.forms.slice(1)
                : details.forms;

            formsList.innerHTML = formsToDisplay.map(form => `
                <div class="word-form-item">
                    <div class="word-form-label">${form.label}:</div>
                    <div class="word-form-value">
                        <span class="word-form-greek">${form.greek}</span>
                        <span class="word-form-russian">${form.russian}</span>
                    </div>
                </div>
            `).join('');
        } else {
            document.getElementById('word-forms-section').classList.add('hidden');
        }
    } catch (error) {
        console.error('Error processing Russian word:', error);
        loading.classList.add('hidden');
        content.style.opacity = '1';
        tg.showAlert('Failed to translate or load word forms');
    }
}

async function showWordDetailsModalForArbitraryWord(greekWord) {
    const modal = document.getElementById('word-details-modal');
    const loading = document.getElementById('word-details-loading');
    const content = document.getElementById('word-details-content');

    // Show modal
    modal.classList.remove('hidden');

    // Set basic info (we don't know translation yet)
    document.getElementById('word-details-title').innerHTML = '🔍 Word Forms <span style="font-size: 12px; font-weight: normal; color: #999;">(not in your list)</span>';
    document.getElementById('word-details-greek').textContent = greekWord;
    document.getElementById('word-details-russian').textContent = '(analyzing...)';
    document.getElementById('word-details-type').textContent = 'analyzing...';

    // Show loading
    loading.classList.remove('hidden');
    content.style.opacity = '0.5';

    try {
        // Fetch word forms from API (for arbitrary word)
        const details = await getWordFormsByText(greekWord);

        // Hide loading
        loading.classList.add('hidden');
        content.style.opacity = '1';

        // Update info with results
        let wordTypeDisplay = details.word_type || 'unknown';

        // Check if first form contains word type info
        if (details.forms && details.forms.length > 0 && details.forms[0].label === 'Word type') {
            wordTypeDisplay = details.forms[0].russian;
            document.getElementById('word-details-russian').textContent = '(analyzing from context)';
        } else {
            document.getElementById('word-details-russian').textContent = details.russian || '(not in your list)';
        }

        document.getElementById('word-details-type').textContent = wordTypeDisplay;

        // Display word forms
        if (details.forms && details.forms.length > 0) {
            const formsSection = document.getElementById('word-forms-section');
            const formsList = document.getElementById('word-forms-list');

            formsSection.classList.remove('hidden');

            // Skip first item if it's just the word type info
            const formsToDisplay = details.forms[0].label === 'Word type'
                ? details.forms.slice(1)
                : details.forms;

            formsList.innerHTML = formsToDisplay.map(form => `
                <div class="word-form-item">
                    <div class="word-form-label">${form.label}:</div>
                    <div class="word-form-value">
                        <span class="word-form-greek">${form.greek}</span>
                        <span class="word-form-russian">${form.russian}</span>
                    </div>
                </div>
            `).join('');
        } else {
            document.getElementById('word-forms-section').classList.add('hidden');
            // Show message if no forms available
            const formsSection = document.getElementById('word-forms-section');
            formsSection.classList.remove('hidden');
            document.getElementById('word-forms-list').innerHTML = `
                <div class="empty-state-text" style="padding: 20px;">
                    Could not determine word forms. This might not be a valid Greek word.
                </div>
            `;
        }
    } catch (error) {
        console.error('Error loading word forms for arbitrary word:', error);
        loading.classList.add('hidden');
        content.style.opacity = '1';
        tg.showAlert('Failed to load word forms');
    }
}

function hideWordDetailsModal() {
    document.getElementById('word-details-modal').classList.add('hidden');
    document.getElementById('word-forms-section').classList.add('hidden');
}

// ========== UI Rendering Functions ==========

function filterWords(words, searchQuery) {
    /**
     * Filter words by search query (searches in both Greek and Russian)
     * @param {Array} words - Array of word objects
     * @param {string} searchQuery - Search query string
     * @returns {Array} Filtered array of words
     */
    if (!searchQuery || searchQuery.trim() === '') {
        return words;
    }

    const query = searchQuery.toLowerCase().trim();

    return words.filter(word => {
        const greekMatch = word.greek.toLowerCase().includes(query);
        const russianMatch = word.russian.toLowerCase().includes(query);
        return greekMatch || russianMatch;
    });
}

function renderLearningWords() {
    const container = document.getElementById('learning-words-list');

    if (state.learningWords.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">📚</div>
                <div class="empty-state-text">No words yet. Fetch some new words!</div>
            </div>
        `;
        document.getElementById('practice-btn').disabled = true;
        return;
    }

    // Filter words by search query
    const filteredWords = filterWords(state.learningWords, state.searchQueries.learning);

    // Show empty state if no words match search
    if (filteredWords.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">🔍</div>
                <div class="empty-state-text">No words found matching your search.</div>
            </div>
        `;
        return;
    }

    document.getElementById('practice-btn').disabled = false;

    container.innerHTML = filteredWords.map(word => {
        const accuracy = word.exercise_count > 0
            ? Math.round((word.correct_count / word.exercise_count) * 100)
            : 0;

        return `
            <div class="word-card" data-word-id="${word.id}">
                <div class="word-content">
                    <div class="word-greek">
                        🇬🇷 ${word.greek}
                        <button class="btn-speak" onclick="playSpeech('${word.greek.replace(/'/g, "\\'")}', 'greek', this)" title="Speak Greek">🔊</button>
                    </div>
                    <div class="word-russian">🇷🇺 ${word.russian}</div>
                    ${word.exercise_count > 0 ? `
                        <div class="word-stats">
                            ${word.correct_count}/${word.exercise_count} correct (${accuracy}%)
                        </div>
                    ` : ''}
                </div>
                <div class="word-actions">
                    <button class="btn btn-secondary" onclick="showEditWordModal('${word.id}')" title="Edit">✎</button>
                    <button class="btn btn-success" onclick="markWordAsLearned('${word.id}')" title="Mark as learned">✓</button>
                    <button class="btn btn-danger" onclick="deleteWord('${word.id}')" title="Delete">✕</button>
                </div>
            </div>
        `;
    }).join('');

    // Add long press handlers to word cards
    attachLongPressHandlers();
}

function renderLearnedWords() {
    const container = document.getElementById('learned-words-list');

    if (state.learnedWords.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">🎓</div>
                <div class="empty-state-text">No learned words yet. Keep practicing!</div>
            </div>
        `;
        return;
    }

    // Filter words by search query
    const filteredWords = filterWords(state.learnedWords, state.searchQueries.learned);

    // Show empty state if no words match search
    if (filteredWords.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">🔍</div>
                <div class="empty-state-text">No words found matching your search.</div>
            </div>
        `;
        return;
    }

    container.innerHTML = filteredWords.map(word => {
        const accuracy = word.total_exercises > 0
            ? Math.round((word.total_correct / word.total_exercises) * 100)
            : 0;

        return `
            <div class="word-card" data-word-id="${word.id}">
                <div class="word-content">
                    <div class="word-greek">
                        🇬🇷 ${word.greek}
                        <button class="btn-speak" onclick="playSpeech('${word.greek.replace(/'/g, "\\'")}', 'greek', this)" title="Speak Greek">🔊</button>
                    </div>
                    <div class="word-russian">🇷🇺 ${word.russian}</div>
                    <div class="word-stats">
                        ${word.total_correct}/${word.total_exercises} correct (${accuracy}%)
                    </div>
                </div>
                <div class="word-actions">
                    <button class="btn btn-secondary" onclick="showEditWordModal('${word.id}')" title="Edit">✎</button>
                    <button class="btn btn-primary" onclick="moveToLearning('${word.id}')" title="Move to learning">↺</button>
                </div>
            </div>
        `;
    }).join('');

    // Add long press handlers to word cards
    attachLongPressHandlers();
}

function renderStats() {
    const container = document.getElementById('stats-content');

    if (state.stats.total_exercises === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">📊</div>
                <div class="empty-state-text">No statistics yet. Start practicing!</div>
            </div>
        `;
        return;
    }

    const grAccuracy = state.stats.greek_to_russian?.total > 0
        ? Math.round((state.stats.greek_to_russian.correct / state.stats.greek_to_russian.total) * 100)
        : 0;

    const rgAccuracy = state.stats.russian_to_greek?.total > 0
        ? Math.round((state.stats.russian_to_greek.correct / state.stats.russian_to_greek.total) * 100)
        : 0;

    container.innerHTML = `
        <div class="stat-card">
            <div class="stat-title">Overall Accuracy</div>
            <div class="stat-value">${Math.round((state.stats.accuracy || 0) * 100)}%</div>
            <div class="stat-details">
                ${state.stats.total_correct}/${state.stats.total_exercises} exercises
            </div>
        </div>

        <div class="stat-card">
            <div class="stat-title">Words Progress</div>
            <div class="stat-value">${state.learningWords.length} / ${state.learnedWords.length}</div>
            <div class="stat-details">
                Learning / Learned
            </div>
        </div>

        <div class="stat-card">
            <div class="stat-title">🇬🇷 → 🇷🇺 Greek to Russian</div>
            <div class="stat-value">${grAccuracy}%</div>
            <div class="stat-details">
                ${state.stats.greek_to_russian?.correct || 0}/${state.stats.greek_to_russian?.total || 0} exercises
            </div>
        </div>

        <div class="stat-card">
            <div class="stat-title">🇷🇺 → 🇬🇷 Russian to Greek</div>
            <div class="stat-value">${rgAccuracy}%</div>
            <div class="stat-details">
                ${state.stats.russian_to_greek?.correct || 0}/${state.stats.russian_to_greek?.total || 0} exercises
            </div>
        </div>

        ${state.stats.streak_days > 0 ? `
            <div class="stat-card">
                <div class="stat-title">🔥 Streak</div>
                <div class="stat-value">${state.stats.streak_days}</div>
                <div class="stat-details">days</div>
            </div>
        ` : ''}
    `;
}

function renderLists() {
    const container = document.getElementById('lists-container');

    if (state.lists.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">📋</div>
                <div class="empty-state-text">No custom lists yet. Create one to organize your words!</div>
            </div>
        `;
        return;
    }

    container.innerHTML = state.lists.map(listName => {
        const stats = getListStats(listName);
        return `
            <div class="list-card" onclick="openManageListModal('${listName.replace(/'/g, "\\'")}')">
                <div class="list-card-header">
                    <div class="list-name">📋 ${listName}</div>
                </div>
                <div class="list-count">
                    ${stats.total} words (${stats.learning} learning, ${stats.learned} learned)
                </div>
            </div>
        `;
    }).join('');
}

function renderManageListModal(listName) {
    const words = getWordsByList(listName);
    const allWords = [...words.learning, ...words.learned];

    // Update title
    document.getElementById('manage-list-title').textContent = `Manage List: ${listName}`;
    document.getElementById('list-word-count').textContent = allWords.length;

    // Render words in list
    const listWordsContainer = document.getElementById('list-words-container');
    if (allWords.length === 0) {
        listWordsContainer.innerHTML = `
            <div class="empty-state-text" style="padding: 20px;">
                No words in this list yet.
            </div>
        `;
    } else {
        listWordsContainer.innerHTML = allWords.map(word => {
            const isLearned = words.learned.includes(word);
            return `
                <div class="word-card">
                    <div class="word-content">
                        <div class="word-greek">🇬🇷 ${word.greek}</div>
                        <div class="word-russian">🇷🇺 ${word.russian}</div>
                        ${isLearned ? '<div class="word-stats">✅ Learned</div>' : ''}
                    </div>
                    <div class="word-actions">
                        <button class="btn btn-danger" onclick="removeWordFromListUI('${word.id}', '${listName.replace(/'/g, "\\'")}')">Remove</button>
                    </div>
                </div>
            `;
        }).join('');
    }

    // Render available words to add
    renderAvailableWordsForList(listName);
}

function renderAvailableWordsForList(listName, searchQuery = '') {
    const words = getWordsByList(listName);
    const wordsInList = new Set([...words.learning, ...words.learned].map(w => w.id));

    // Get all words not in list
    let availableWords = [
        ...state.learningWords.filter(w => !wordsInList.has(w.id)),
        ...state.learnedWords.filter(w => !wordsInList.has(w.id))
    ];

    // Filter by search
    if (searchQuery) {
        const query = searchQuery.toLowerCase();
        availableWords = availableWords.filter(w =>
            w.greek.toLowerCase().includes(query) ||
            w.russian.toLowerCase().includes(query)
        );
    }

    const container = document.getElementById('available-words-container');

    if (availableWords.length === 0) {
        container.innerHTML = `
            <div class="empty-state-text" style="padding: 20px;">
                ${searchQuery ? 'No matching words found.' : 'All words are already in this list.'}
            </div>
        `;
        return;
    }

    container.innerHTML = availableWords.slice(0, 20).map(word => {
        return `
            <div class="word-card">
                <div class="word-content">
                    <div class="word-greek">🇬🇷 ${word.greek}</div>
                    <div class="word-russian">🇷🇺 ${word.russian}</div>
                </div>
                <div class="word-actions">
                    <button class="btn btn-success" onclick="addWordToListUI('${word.id}', '${listName.replace(/'/g, "\\'")}')">+ Add</button>
                </div>
            </div>
        `;
    }).join('');

    if (availableWords.length > 20) {
        container.innerHTML += `
            <div class="empty-state-text" style="padding: 10px;">
                Showing first 20 words. Use search to find more.
            </div>
        `;
    }
}

function renderMatchingExercise() {
    const ex = state.currentExercise;

    document.getElementById('exercise-loading').classList.add('hidden');
    document.getElementById('exercise-content').style.display = 'block';

    // Update counter (with list name if practicing a list)
    let counterText = `Exercise ${state.exerciseCount}`;
    if (state.selectedListForPractice) {
        counterText += ` - 📋 ${state.selectedListForPractice}`;
    }
    document.getElementById('exercise-counter').textContent = counterText;

    // Prepare pairs for matching
    const pairs = ex.pairs;
    const direction = ex.direction;

    // Shuffle left and right columns
    const leftItems = pairs.map(p => ({
        id: p.word_id,
        text: direction === 'greek_to_russian' ? p.greek : p.russian
    }));

    const rightItems = [...pairs].map(p => ({
        id: p.word_id,
        text: direction === 'greek_to_russian' ? p.russian : p.greek
    })).sort(() => Math.random() - 0.5);

    // Build HTML
    const questionText = direction === 'greek_to_russian'
        ? 'Match Greek words with Russian translations:'
        : 'Match Russian words with Greek translations:';

    document.getElementById('exercise-question').innerHTML = `
        ${questionText}
        <div class="exercise-hint">
            💡 Long press any card to see word forms
            <span style="margin-left: 10px;">🔊 Click speaker icon to hear pronunciation</span>
        </div>
    `;
    document.getElementById('exercise-sentence').style.display = 'none';
    document.getElementById('exercise-sentence-clickable').style.display = 'none';
    document.getElementById('exercise-translation').style.display = 'none';
    document.getElementById('exercise-options').style.display = 'none';
    document.getElementById('exercise-feedback').classList.add('hidden');
    document.getElementById('next-exercise-btn').classList.add('hidden');
    document.getElementById('mark-learned-btn').classList.add('hidden');

    // Create matching container
    const contentDiv = document.getElementById('exercise-content');
    const leftLanguage = direction === 'greek_to_russian' ? 'greek' : 'russian';
    const rightLanguage = direction === 'greek_to_russian' ? 'russian' : 'greek';

    let matchingHTML = `
        <div class="matching-progress">
            Matched: <span id="matched-count">0</span> / ${pairs.length}
        </div>
        <div class="matching-container">
            <div class="matching-column" id="left-column">
                ${leftItems.map(item => `
                    <div class="matching-card" data-id="${item.id}" data-side="left">
                        <div class="matching-card-text">${item.text}</div>
                        <button class="btn-speak-small" onclick="event.stopPropagation(); playSpeech('${item.text.replace(/'/g, "\\'")}', '${leftLanguage}', this)" title="Speak">🔊</button>
                    </div>
                `).join('')}
            </div>
            <div class="matching-column" id="right-column">
                ${rightItems.map(item => `
                    <div class="matching-card" data-id="${item.id}" data-side="right">
                        <div class="matching-card-text">${item.text}</div>
                        <button class="btn-speak-small" onclick="event.stopPropagation(); playSpeech('${item.text.replace(/'/g, "\\'")}', '${rightLanguage}', this)" title="Speak">🔊</button>
                    </div>
                `).join('')}
            </div>
        </div>
    `;

    // Find or create matching container
    let matchingContainer = document.getElementById('matching-exercise-container');
    if (!matchingContainer) {
        matchingContainer = document.createElement('div');
        matchingContainer.id = 'matching-exercise-container';
        const question = document.getElementById('exercise-question');
        question.parentNode.insertBefore(matchingContainer, question.nextSibling);
    }
    matchingContainer.innerHTML = matchingHTML;

    // Add click handlers
    document.querySelectorAll('.matching-card').forEach(card => {
        card.addEventListener('click', () => handleMatchingCardClick(card));
    });

    // Add long press handlers to matching cards
    attachLongPressToMatchingCards();

    showExerciseScreen();
}

function handleMatchingCardClick(card) {
    if (card.classList.contains('matched')) return;

    const side = card.dataset.side;
    const cardId = card.dataset.id;

    // Deselect previous selection on the same side
    if (side === 'left') {
        if (state.matchingState.selectedLeft) {
            state.matchingState.selectedLeft.classList.remove('selected');
        }
        state.matchingState.selectedLeft = card;
        card.classList.add('selected');
    } else {
        if (state.matchingState.selectedRight) {
            state.matchingState.selectedRight.classList.remove('selected');
        }
        state.matchingState.selectedRight = card;
        card.classList.add('selected');
    }

    // Check if both sides are selected
    if (state.matchingState.selectedLeft && state.matchingState.selectedRight) {
        const leftId = state.matchingState.selectedLeft.dataset.id;
        const rightId = state.matchingState.selectedRight.dataset.id;

        if (leftId === rightId) {
            // Correct match
            state.matchingState.selectedLeft.classList.add('matched');
            state.matchingState.selectedRight.classList.add('matched');
            state.matchingState.selectedLeft.classList.remove('selected');
            state.matchingState.selectedRight.classList.remove('selected');

            state.matchingState.matchedPairs.push({
                word_id: leftId,
                is_correct: true
            });

            // Haptic feedback
            if (tg.HapticFeedback) {
                tg.HapticFeedback.notificationOccurred('success');
            }

            // Update progress
            document.getElementById('matched-count').textContent = state.matchingState.matchedPairs.length;

            // Check if all pairs matched
            if (state.matchingState.matchedPairs.length === state.currentExercise.pairs.length) {
                setTimeout(() => showMatchingComplete(), 500);
            }

            state.matchingState.selectedLeft = null;
            state.matchingState.selectedRight = null;
        } else {
            // Incorrect match
            state.matchingState.selectedLeft.classList.add('incorrect');
            state.matchingState.selectedRight.classList.add('incorrect');
            state.matchingState.incorrectAttempts++;

            // Haptic feedback
            if (tg.HapticFeedback) {
                tg.HapticFeedback.notificationOccurred('error');
            }

            // Remove incorrect styling after animation
            setTimeout(() => {
                if (state.matchingState.selectedLeft) {
                    state.matchingState.selectedLeft.classList.remove('incorrect', 'selected');
                }
                if (state.matchingState.selectedRight) {
                    state.matchingState.selectedRight.classList.remove('incorrect', 'selected');
                }
                state.matchingState.selectedLeft = null;
                state.matchingState.selectedRight = null;
            }, 500);
        }
    }
}

async function showMatchingComplete() {
    const totalPairs = state.currentExercise.pairs.length;
    const accuracy = Math.round((totalPairs / (totalPairs + state.matchingState.incorrectAttempts)) * 100);

    // Send results to backend
    await validateMatchingResults(state.matchingState.matchedPairs);

    // Show completion screen
    const matchingContainer = document.getElementById('matching-exercise-container');
    matchingContainer.innerHTML = `
        <div class="matching-complete">
            <div class="matching-complete-icon">🎉</div>
            <div class="matching-complete-text">All pairs matched!</div>
            <div class="matching-complete-stats">
                Accuracy: ${accuracy}%<br>
                Incorrect attempts: ${state.matchingState.incorrectAttempts}
            </div>
        </div>
    `;

    document.getElementById('next-exercise-btn').classList.remove('hidden');
}

function parseSentenceWords(sentence, correctAnswer, wordId) {
    /**
     * Parse sentence into clickable word elements.
     * Handles multi-word phrases by grouping them into a single button.
     *
     * @param {string} sentence - The sentence to parse
     * @param {string} correctAnswer - The correct answer (may be multi-word)
     * @param {string} wordId - The word ID for the correct answer
     * @returns {string} HTML string of clickable word elements
     */
    const words = sentence.split(/\s+/); // Split by whitespace

    // Check if correct answer is a multi-word phrase
    const correctAnswerWords = correctAnswer.split(/\s+/).map(word =>
        word.replace(/[.,!?;:»«"""'()—–\-\[\]]/g, '').toLowerCase()
    );
    const isMultiWord = correctAnswerWords.length > 1;

    // If multi-word, find the starting index in the sentence
    let multiWordStartIndex = -1;
    if (isMultiWord) {
        for (let i = 0; i <= words.length - correctAnswerWords.length; i++) {
            let match = true;
            for (let j = 0; j < correctAnswerWords.length; j++) {
                const cleanWord = words[i + j].replace(/[.,!?;:»«"""'()—–\-\[\]]/g, '').toLowerCase();
                if (cleanWord !== correctAnswerWords[j]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                multiWordStartIndex = i;
                break;
            }
        }
    }

    const htmlParts = words.map((word, index) => {
        // Skip words that are part of a multi-word phrase (but not the first word)
        if (isMultiWord && multiWordStartIndex !== -1 &&
            index > multiWordStartIndex && index < multiWordStartIndex + correctAnswerWords.length) {
            return ''; // Skip this word, it's included in the multi-word button
        }

        // Clean up punctuation for comparison but keep it for display
        const cleanWord = word.replace(/[.,!?;:»«"""'()—–\-\[\]]/g, '');

        // Check if this is the start of a multi-word phrase
        if (isMultiWord && index === multiWordStartIndex) {
            // Create a button for the entire multi-word phrase
            const phraseWords = words.slice(index, index + correctAnswerWords.length);
            const phraseText = phraseWords.join(' ');
            const phraseClean = correctAnswer; // Use the correct answer for comparison
            const wordIdAttr = `data-word-id="${wordId}"`;
            return `<span class="sentence-word" data-word="${phraseClean}" data-index="${index}" ${wordIdAttr}>${phraseText}</span>`;
        }

        // Regular single word
        const isTargetWord = cleanWord.toLowerCase() === correctAnswer.toLowerCase();
        const wordIdAttr = isTargetWord ? `data-word-id="${wordId}"` : '';
        return `<span class="sentence-word" data-word="${cleanWord}" data-index="${index}" ${wordIdAttr}>${word}</span>`;
    }).filter(html => html !== '');

    return htmlParts.join('');
}

function renderExercise() {
    const ex = state.currentExercise;

    // Make sure content is visible and loading is hidden
    document.getElementById('exercise-loading').classList.add('hidden');
    document.getElementById('exercise-content').style.display = 'block';

    // Update counter (with list name if practicing a list)
    let counterText = `Exercise ${state.exerciseCount}`;
    if (state.selectedListForPractice) {
        counterText += ` - 📋 ${state.selectedListForPractice}`;
    }
    document.getElementById('exercise-counter').textContent = counterText;

    // Build question text
    let questionText = '';
    if (ex.direction === 'greek_to_russian') {
        questionText = 'Click on the Greek word that means:';
    } else {
        questionText = 'Click on the Russian word that means:';
    }

    // Determine what to ask for
    let targetWord = '';
    if (ex.direction === 'greek_to_russian') {
        targetWord = ex.translated_word; // Russian word
    } else {
        targetWord = ex.translated_word; // Greek word
    }

    // Determine language for speak button
    const sentenceLanguage = ex.direction === 'greek_to_russian' ? 'greek' : 'russian';
    const translationLanguage = ex.direction === 'greek_to_russian' ? 'russian' : 'greek';

    document.getElementById('exercise-question').innerHTML = `
        ${questionText}<br><strong>"${targetWord}"</strong>
        <div class="exercise-hint">
            💡 Long press any word to see its forms (even if not in your list)
            <button class="btn-speak" onclick="playSpeech('${ex.sentence.replace(/'/g, "\\'")}', '${sentenceLanguage}', this)" title="Speak sentence">🔊 Play sentence</button>
        </div>
    `;

    // Hide the non-interactive sentence, show clickable version
    document.getElementById('exercise-sentence').style.display = 'none';

    // Hide matching container if exists
    const matchingContainer = document.getElementById('matching-exercise-container');
    if (matchingContainer) {
        matchingContainer.innerHTML = '';
    }

    // Show sentence clickable
    document.getElementById('exercise-sentence-clickable').style.display = 'flex';
    document.getElementById('exercise-translation').style.display = 'block';

    // Render clickable sentence words
    const clickableContainer = document.getElementById('exercise-sentence-clickable');
    clickableContainer.innerHTML = parseSentenceWords(ex.sentence, ex.correct_answer, ex.word_id);

    // Add click handlers to words
    document.querySelectorAll('.sentence-word').forEach(wordEl => {
        wordEl.addEventListener('click', () => handleWordClick(wordEl));
    });

    // Add long press handlers to words
    attachLongPressToSentenceWords();

    // Display translation with speak button
    document.getElementById('exercise-translation').innerHTML = `
        <span>Translation: ${ex.sentence_translation}</span>
        <button class="btn-speak" onclick="playSpeech('${ex.sentence_translation.replace(/'/g, "\\'")}', '${translationLanguage}', this)" title="Speak translation">🔊</button>
    `;

    // Hide button options, hide feedback and buttons
    document.getElementById('exercise-options').style.display = 'none';
    document.getElementById('exercise-feedback').classList.add('hidden');
    document.getElementById('next-exercise-btn').classList.add('hidden');
    document.getElementById('mark-learned-btn').classList.add('hidden');

    showExerciseScreen();
}

async function handleWordClick(wordElement) {
    // Disable all word elements
    const allWords = document.querySelectorAll('.sentence-word');
    allWords.forEach(word => word.classList.add('disabled'));

    const selectedWord = wordElement.dataset.word;

    try {
        const result = await validateAnswer(selectedWord);

        // Highlight correct/incorrect
        if (result.correct) {
            wordElement.classList.add('correct');
        } else {
            wordElement.classList.add('incorrect');
            // Also highlight the correct word (normalize for comparison)
            const normalizeForComparison = (text) => {
                return text.replace(/[.,!?;:»«"""'()—–\-\[\]]/g, '').trim().toLowerCase();
            };
            const correctNormalized = normalizeForComparison(result.correct_answer);
            allWords.forEach(word => {
                if (normalizeForComparison(word.dataset.word) === correctNormalized) {
                    word.classList.add('correct');
                }
            });
        }

        // Show feedback
        const feedback = document.getElementById('exercise-feedback');
        feedback.textContent = result.explanation;
        feedback.className = 'feedback ' + (result.correct ? 'success' : 'error');

        // Show next button
        document.getElementById('next-exercise-btn').classList.remove('hidden');
        document.getElementById('mark-learned-btn').classList.remove('hidden');

        // Vibrate on correct answer
        if (result.correct && tg.HapticFeedback) {
            tg.HapticFeedback.notificationOccurred('success');
        }

    } catch (error) {
        console.error('Error validating answer:', error);
        tg.showAlert('Failed to validate answer');
        allWords.forEach(word => word.classList.remove('disabled'));
    }
}

async function handleOptionClick(selectedOption) {
    const optionButtons = document.querySelectorAll('.option-btn');
    optionButtons.forEach(btn => btn.disabled = true);

    try {
        const result = await validateAnswer(selectedOption);

        // Highlight correct/incorrect
        optionButtons.forEach(btn => {
            if (btn.textContent === result.correct_answer) {
                btn.classList.add('correct');
            } else if (btn.textContent === selectedOption && !result.correct) {
                btn.classList.add('incorrect');
            }
        });

        // Show feedback
        const feedback = document.getElementById('exercise-feedback');
        feedback.textContent = result.explanation;
        feedback.className = 'feedback ' + (result.correct ? 'success' : 'error');

        // Show next button
        document.getElementById('next-exercise-btn').classList.remove('hidden');
        document.getElementById('mark-learned-btn').classList.remove('hidden');

        // Vibrate on correct answer
        if (result.correct && tg.HapticFeedback) {
            tg.HapticFeedback.notificationOccurred('success');
        }

    } catch (error) {
        console.error('Error validating answer:', error);
        tg.showAlert('Failed to validate answer');
        optionButtons.forEach(btn => btn.disabled = false);
    }
}

// ========== Screen Management ==========

function showScreen(screenId) {
    document.querySelectorAll('.screen').forEach(screen => {
        screen.classList.remove('active');
    });
    document.getElementById(screenId).classList.add('active');
}

function showLoading() {
    showScreen('loading-screen');
}

function hideLoading() {
    showScreen('main-screen');
}

function showMainScreen() {
    showScreen('main-screen');
    // Reset list practice mode
    state.selectedListForPractice = null;
}

function showExerciseScreen() {
    showScreen('exercise-screen');
}

function showExerciseLoading() {
    showScreen('exercise-screen');
    document.getElementById('exercise-loading').classList.remove('hidden');
    document.getElementById('exercise-content').style.display = 'none';
}

function hideExerciseLoading() {
    document.getElementById('exercise-loading').classList.add('hidden');
    document.getElementById('exercise-content').style.display = 'block';
}

// ========== Tab Management ==========

function switchTab(tabName) {
    // Update tab buttons
    document.querySelectorAll('.tab').forEach(tab => {
        tab.classList.remove('active');
    });
    document.querySelector(`.tab[data-tab="${tabName}"]`).classList.add('active');

    // Update tab panels
    document.querySelectorAll('.tab-panel').forEach(panel => {
        panel.classList.remove('active');
    });
    document.getElementById(`${tabName}-tab`).classList.add('active');

    // Load data for the tab
    if (tabName === 'learning') {
        loadWords();
    } else if (tabName === 'learned') {
        loadLearnedWords();
    } else if (tabName === 'lists') {
        state.lists = getAllListNames();
        renderLists();
    } else if (tabName === 'stats') {
        loadStats();
    }
}

// ========== Modal Management ==========

function showFetchWordsModal() {
    console.log('showFetchWordsModal called');
    const modal = document.getElementById('fetch-words-modal');
    console.log('Modal element:', modal);
    console.log('Modal classList before:', modal.classList.toString());
    modal.classList.remove('hidden');
    console.log('Modal classList after:', modal.classList.toString());
}

function hideFetchWordsModal() {
    document.getElementById('fetch-words-modal').classList.add('hidden');
}

function showExerciseTypeModal() {
    // Check if practicing a list or all words
    let wordsToCheck = state.learningWords;
    if (state.selectedListForPractice) {
        const listWords = getWordsByList(state.selectedListForPractice);
        wordsToCheck = listWords.learning;
    }

    // Check if user has at least 10 words for matching exercise
    const hasEnoughWords = wordsToCheck.length >= 10;

    if (!hasEnoughWords) {
        document.getElementById('matching-exercise-btn').disabled = true;
        document.getElementById('matching-exercise-btn').style.opacity = '0.5';
    } else {
        document.getElementById('matching-exercise-btn').disabled = false;
        document.getElementById('matching-exercise-btn').style.opacity = '1';
    }

    document.getElementById('exercise-type-modal').classList.remove('hidden');
}

function hideExerciseTypeModal() {
    document.getElementById('exercise-type-modal').classList.add('hidden');
}

function showAddCustomWordModal() {
    const modal = document.getElementById('add-custom-word-modal');
    const input = document.getElementById('custom-word-input');
    const errorEl = document.getElementById('add-word-error');
    const loadingEl = document.getElementById('add-word-loading');

    // Reset modal state
    input.value = '';
    errorEl.classList.add('hidden');
    loadingEl.classList.add('hidden');

    // Show modal
    modal.classList.remove('hidden');

    // Focus input
    setTimeout(() => input.focus(), 100);
}

function hideAddCustomWordModal() {
    document.getElementById('add-custom-word-modal').classList.add('hidden');
}

function showCreateListModal() {
    const modal = document.getElementById('create-list-modal');
    const input = document.getElementById('list-name-input');

    input.value = '';
    modal.classList.remove('hidden');
    setTimeout(() => input.focus(), 100);
}

function hideCreateListModal() {
    document.getElementById('create-list-modal').classList.add('hidden');
}

async function createListFromModal() {
    const input = document.getElementById('list-name-input');
    const listName = input.value.trim();

    if (!listName) {
        tg.showAlert('Please enter a list name');
        return;
    }

    if (state.lists.includes(listName)) {
        tg.showAlert('A list with this name already exists');
        return;
    }

    // Just add to state - list will be created when first word is added
    state.lists.push(listName);
    state.lists.sort();

    hideCreateListModal();
    renderLists();
    tg.showAlert(`List "${listName}" created!`);
}

function openManageListModal(listName) {
    state.currentList = listName;

    const modal = document.getElementById('manage-list-modal');
    modal.classList.remove('hidden');

    renderManageListModal(listName);
}

function hideManageListModal() {
    document.getElementById('manage-list-modal').classList.add('hidden');
    state.currentList = null;
}

async function addWordToListUI(wordId, listName) {
    const success = await addWordToList(wordId, listName);

    if (success) {
        // Refresh modal view
        renderManageListModal(listName);
    } else {
        tg.showAlert('Failed to add word to list');
    }
}

async function removeWordFromListUI(wordId, listName) {
    const success = await removeWordFromList(wordId, listName);

    if (success) {
        // Refresh modal view
        renderManageListModal(listName);

        // If list is now empty, remove it from state
        const stats = getListStats(listName);
        if (stats.total === 0) {
            state.lists = state.lists.filter(l => l !== listName);
            hideManageListModal();
            renderLists();
        }
    } else {
        tg.showAlert('Failed to remove word from list');
    }
}

async function renameListUI() {
    if (!state.currentList) return;

    const newName = prompt(`Enter new name for list "${state.currentList}":`, state.currentList);

    if (!newName || newName.trim() === '') {
        return;
    }

    const trimmedName = newName.trim();

    if (trimmedName === state.currentList) {
        return;
    }

    if (state.lists.includes(trimmedName)) {
        tg.showAlert('A list with this name already exists');
        return;
    }

    const success = await renameList(state.currentList, trimmedName);

    if (success) {
        tg.showAlert(`List renamed to "${trimmedName}"`);
        state.currentList = trimmedName;
        hideManageListModal();
        renderLists();
    } else {
        tg.showAlert('Failed to rename list');
    }
}

async function deleteListUI() {
    if (!state.currentList) return;

    const confirmed = confirm(`Are you sure you want to delete the list "${state.currentList}"? Words will not be deleted, only removed from this list.`);

    if (!confirmed) return;

    const success = await deleteList(state.currentList);

    if (success) {
        tg.showAlert('List deleted');
        hideManageListModal();
        renderLists();
    } else {
        tg.showAlert('Failed to delete list');
    }
}

function practiceListUI() {
    if (!state.currentList) return;

    const stats = getListStats(state.currentList);

    if (stats.total === 0) {
        tg.showAlert('This list has no words');
        return;
    }

    state.selectedListForPractice = state.currentList;
    hideManageListModal();
    showExerciseTypeModal();
}

function showEditWordModal(wordId) {
    // Find word in learning or learned
    let word = state.learningWords.find(w => w.id === wordId);
    if (!word) {
        word = state.learnedWords.find(w => w.id === wordId);
    }

    if (!word) {
        console.error('Word not found:', wordId);
        return;
    }

    // Populate form
    document.getElementById('edit-word-greek').value = word.greek;
    document.getElementById('edit-word-russian').value = word.russian;
    document.getElementById('edit-word-type').value = word.word_type || '';
    document.getElementById('edit-word-error').classList.add('hidden');

    // Store word ID in modal for later use
    const modal = document.getElementById('edit-word-modal');
    modal.dataset.wordId = wordId;
    modal.classList.remove('hidden');

    // Focus first input
    setTimeout(() => document.getElementById('edit-word-greek').focus(), 100);
}

function hideEditWordModal() {
    const modal = document.getElementById('edit-word-modal');
    modal.classList.add('hidden');
    delete modal.dataset.wordId;
}

async function saveEditedWord() {
    const modal = document.getElementById('edit-word-modal');
    const wordId = modal.dataset.wordId;

    if (!wordId) {
        console.error('No word ID in modal');
        return;
    }

    const greek = document.getElementById('edit-word-greek').value.trim();
    const russian = document.getElementById('edit-word-russian').value.trim();
    const wordType = document.getElementById('edit-word-type').value;
    const errorEl = document.getElementById('edit-word-error');

    // Validate
    if (!greek || !russian) {
        errorEl.textContent = 'Both Greek and Russian fields are required';
        errorEl.classList.remove('hidden');
        return;
    }

    try {
        // Update word
        const updates = {
            greek: greek,
            russian: russian,
            word_type: wordType
        };

        await editWord(wordId, updates);

        // Reload words
        await loadWords();
        await loadLearnedWords();

        // Update lists if on lists tab
        if (state.currentList) {
            renderManageListModal(state.currentList);
        }

        hideEditWordModal();
        tg.showAlert('Word updated successfully!');
    } catch (error) {
        console.error('Error saving word:', error);
        errorEl.textContent = error.message || 'Failed to save word';
        errorEl.classList.remove('hidden');
    }
}

// ========== Event Listeners ==========

document.addEventListener('DOMContentLoaded', () => {
    // Tab switching
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', () => {
            switchTab(tab.dataset.tab);
        });
    });

    // Fetch words button - use event delegation to handle dynamic content
    document.addEventListener('click', (e) => {
        if (e.target.id === 'fetch-words-btn' || e.target.closest('#fetch-words-btn')) {
            console.log('Fetch words button clicked!');
            e.preventDefault();
            e.stopPropagation();
            showFetchWordsModal();
        }
    });

    // Fetch words modal
    document.getElementById('fetch-confirm-btn').addEventListener('click', async () => {
        const count = parseInt(document.getElementById('fetch-count-input').value);
        if (count >= 1 && count <= 600) {
            hideFetchWordsModal();
            await fetchNewWords(count);
        }
    });

    document.getElementById('fetch-cancel-btn').addEventListener('click', hideFetchWordsModal);

    // Add custom word button - use event delegation
    document.addEventListener('click', (e) => {
        if (e.target.id === 'add-custom-word-btn' || e.target.closest('#add-custom-word-btn')) {
            console.log('Add custom word button clicked!');
            e.preventDefault();
            e.stopPropagation();
            showAddCustomWordModal();
        }
    });

    // Add custom word modal
    document.getElementById('add-custom-word-confirm-btn').addEventListener('click', async () => {
        const wordText = document.getElementById('custom-word-input').value.trim();
        if (wordText) {
            await addCustomWord(wordText);
        }
    });

    document.getElementById('add-custom-word-cancel-btn').addEventListener('click', hideAddCustomWordModal);

    // Allow Enter key to submit
    document.getElementById('custom-word-input').addEventListener('keypress', async (e) => {
        if (e.key === 'Enter') {
            const wordText = e.target.value.trim();
            if (wordText) {
                await addCustomWord(wordText);
            }
        }
    });

    // Close modals when clicking outside
    document.getElementById('fetch-words-modal').addEventListener('click', (e) => {
        if (e.target.id === 'fetch-words-modal') {
            hideFetchWordsModal();
        }
    });

    document.getElementById('exercise-type-modal').addEventListener('click', (e) => {
        if (e.target.id === 'exercise-type-modal') {
            hideExerciseTypeModal();
        }
    });

    document.getElementById('word-details-modal').addEventListener('click', (e) => {
        if (e.target.id === 'word-details-modal') {
            hideWordDetailsModal();
        }
    });

    document.getElementById('word-details-close').addEventListener('click', hideWordDetailsModal);

    document.getElementById('add-custom-word-modal').addEventListener('click', (e) => {
        if (e.target.id === 'add-custom-word-modal') {
            hideAddCustomWordModal();
        }
    });

    // Word type filter dropdown
    const wordTypeSelect = document.getElementById('word-type-select');
    if (wordTypeSelect) {
        wordTypeSelect.addEventListener('change', () => {
            updateSelectedWordType();
        });
    }

    // Search inputs
    const learningSearchInput = document.getElementById('learning-search-input');
    if (learningSearchInput) {
        learningSearchInput.addEventListener('input', (e) => {
            state.searchQueries.learning = e.target.value;
            renderLearningWords();
        });
    }

    const learnedSearchInput = document.getElementById('learned-search-input');
    if (learnedSearchInput) {
        learnedSearchInput.addEventListener('input', (e) => {
            state.searchQueries.learned = e.target.value;
            renderLearnedWords();
        });
    }

    // Lists management
    document.getElementById('create-list-btn').addEventListener('click', showCreateListModal);

    document.getElementById('create-list-cancel-btn').addEventListener('click', hideCreateListModal);

    document.getElementById('create-list-confirm-btn').addEventListener('click', createListFromModal);

    document.getElementById('list-name-input').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            createListFromModal();
        }
    });

    document.getElementById('manage-list-close').addEventListener('click', hideManageListModal);

    document.getElementById('manage-list-modal').addEventListener('click', (e) => {
        if (e.target.id === 'manage-list-modal') {
            hideManageListModal();
        }
    });

    document.getElementById('rename-list-btn').addEventListener('click', renameListUI);

    document.getElementById('delete-list-btn').addEventListener('click', deleteListUI);

    document.getElementById('practice-list-btn').addEventListener('click', practiceListUI);

    document.getElementById('create-list-modal').addEventListener('click', (e) => {
        if (e.target.id === 'create-list-modal') {
            hideCreateListModal();
        }
    });

    const addToListSearch = document.getElementById('add-to-list-search');
    if (addToListSearch) {
        addToListSearch.addEventListener('input', (e) => {
            if (state.currentList) {
                renderAvailableWordsForList(state.currentList, e.target.value);
            }
        });
    }

    // Edit word modal
    document.getElementById('edit-word-cancel-btn').addEventListener('click', hideEditWordModal);

    document.getElementById('edit-word-save-btn').addEventListener('click', saveEditedWord);

    document.getElementById('edit-word-modal').addEventListener('click', (e) => {
        if (e.target.id === 'edit-word-modal') {
            hideEditWordModal();
        }
    });

    // Allow Enter to save in edit modal
    ['edit-word-greek', 'edit-word-russian'].forEach(inputId => {
        document.getElementById(inputId).addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                saveEditedWord();
            }
        });
    });

    // Exercise type modal
    document.getElementById('sentence-exercise-btn').addEventListener('click', () => {
        hideExerciseTypeModal();
        state.exerciseCount = 0;
        state.exerciseType = 'sentence_context';
        getExercise();
    });

    document.getElementById('matching-exercise-btn').addEventListener('click', () => {
        if (state.learningWords.length >= 10) {
            hideExerciseTypeModal();
            state.exerciseCount = 0;
            state.exerciseType = 'matching_cards';
            getMatchingExercise();
        } else {
            tg.showAlert('You need at least 10 words to play matching cards');
        }
    });

    document.getElementById('exercise-type-cancel-btn').addEventListener('click', hideExerciseTypeModal);

    // Practice button - use event delegation
    document.addEventListener('click', (e) => {
        if (e.target.id === 'practice-btn' || e.target.closest('#practice-btn')) {
            if (state.learningWords.length > 0) {
                e.preventDefault();
                e.stopPropagation();
                // Reset list practice mode when practicing from main screen
                state.selectedListForPractice = null;
                showExerciseTypeModal();
            }
        }
    });

    // Exercise screen buttons
    document.getElementById('exit-exercise-btn').addEventListener('click', showMainScreen);

    document.getElementById('next-exercise-btn').addEventListener('click', () => {
        if (state.exerciseType === 'matching_cards') {
            getMatchingExercise();
        } else {
            getExercise();
        }
    });

    document.getElementById('mark-learned-btn').addEventListener('click', async () => {
        await markWordAsLearned(state.currentExercise.word_id);

        if (state.learningWords.length > 0) {
            getExercise();
        } else {
            tg.showAlert('All words learned! 🎉');
            showMainScreen();
        }
    });

    // Initialize app
    initApp();
});

// ========== Initialization ==========

async function initApp() {
    console.log('=== Initializing Greek Learning App ===');
    showLoading();

    // Check API health first
    try {
        console.log('Checking API health...');
        const healthResponse = await fetch('/greek/api/health');
        const health = await healthResponse.json();
        console.log('API Health:', health);
    } catch (error) {
        console.error('API health check failed:', error);
        hideLoading();
        document.getElementById('main-screen').innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">🔌</div>
                <div class="empty-state-text">
                    Cannot connect to API server.<br>
                    Please make sure the server is running.
                </div>
                <button onclick="initApp()" class="btn btn-primary mt-20">Retry</button>
            </div>
        `;
        return;
    }

    // Check if Telegram WebApp is available
    if (!tg || !tg.initData) {
        console.error('Telegram WebApp not available or initData missing');
        hideLoading();
        document.getElementById('main-screen').innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">⚠️</div>
                <div class="empty-state-text">
                    This app must be opened from Telegram.<br>
                    Please use the bot to open the Web App.
                </div>
            </div>
        `;
        return;
    }

    try {
        console.log('Loading initial data...');
        await Promise.all([
            loadWords().catch(e => {
                console.error('Failed to load words:', e);
                return null;
            }),
            loadLearnedWords().catch(e => {
                console.error('Failed to load learned words:', e);
                return null;
            }),
            loadStats().catch(e => {
                console.error('Failed to load stats:', e);
                return null;
            })
        ]);

        console.log('App initialized successfully');
        hideLoading();

    } catch (error) {
        console.error('Error initializing app:', error);
        hideLoading();

        // Show error message in UI
        document.getElementById('main-screen').innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">❌</div>
                <div class="empty-state-text">
                    Failed to load app: ${error.message}<br>
                    Please check console for details.
                </div>
                <button onclick="initApp()" class="btn btn-primary mt-20">Retry</button>
            </div>
        `;

        if (tg?.showAlert) {
            tg.showAlert('Failed to initialize app: ' + error.message);
        }
    }
}

// Make initApp available globally for retry button
window.initApp = initApp;

// Make functions available globally
window.deleteWord = deleteWord;
window.markWordAsLearned = markWordAsLearned;
window.moveToLearning = moveToLearning;
window.handleOptionClick = handleOptionClick;
window.openManageListModal = openManageListModal;
window.addWordToListUI = addWordToListUI;
window.removeWordFromListUI = removeWordFromListUI;
window.showEditWordModal = showEditWordModal;
window.playSpeech = playSpeech;
