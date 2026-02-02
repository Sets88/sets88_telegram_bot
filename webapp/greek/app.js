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
    matchingState: {
        selectedLeft: null,
        selectedRight: null,
        matchedPairs: [],
        incorrectAttempts: 0
    }
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
    }
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

async function getExercise(wordId = null, direction = 'random') {
    try {
        // Show loading spinner
        showExerciseLoading();

        let url = 'exercise?';
        if (wordId) url += `word_id=${wordId}&`;
        url += `direction=${direction}`;

        // Add word type filter if selected
        if (state.selectedWordType) {
            url += `&word_type=${state.selectedWordType}`;
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

// ========== UI Rendering Functions ==========

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

    document.getElementById('practice-btn').disabled = false;

    container.innerHTML = state.learningWords.map(word => {
        const accuracy = word.exercise_count > 0
            ? Math.round((word.correct_count / word.exercise_count) * 100)
            : 0;

        return `
            <div class="word-card">
                <div class="word-content">
                    <div class="word-greek">🇬🇷 ${word.greek}</div>
                    <div class="word-russian">🇷🇺 ${word.russian}</div>
                    ${word.exercise_count > 0 ? `
                        <div class="word-stats">
                            ${word.correct_count}/${word.exercise_count} correct (${accuracy}%)
                        </div>
                    ` : ''}
                </div>
                <div class="word-actions">
                    <button class="btn btn-success" onclick="markWordAsLearned('${word.id}')">✓</button>
                    <button class="btn btn-danger" onclick="deleteWord('${word.id}')">✕</button>
                </div>
            </div>
        `;
    }).join('');
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

    container.innerHTML = state.learnedWords.map(word => {
        const accuracy = word.total_exercises > 0
            ? Math.round((word.total_correct / word.total_exercises) * 100)
            : 0;

        return `
            <div class="word-card">
                <div class="word-content">
                    <div class="word-greek">🇬🇷 ${word.greek}</div>
                    <div class="word-russian">🇷🇺 ${word.russian}</div>
                    <div class="word-stats">
                        ${word.total_correct}/${word.total_exercises} correct (${accuracy}%)
                    </div>
                </div>
                <div class="word-actions">
                    <button class="btn btn-primary" onclick="moveToLearning('${word.id}')">↺</button>
                </div>
            </div>
        `;
    }).join('');
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

function renderMatchingExercise() {
    const ex = state.currentExercise;

    document.getElementById('exercise-loading').classList.add('hidden');
    document.getElementById('exercise-content').style.display = 'block';
    document.getElementById('exercise-counter').textContent = `Exercise ${state.exerciseCount}`;

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

    document.getElementById('exercise-question').innerHTML = questionText;
    document.getElementById('exercise-sentence').style.display = 'none';
    document.getElementById('exercise-sentence-clickable').style.display = 'none';
    document.getElementById('exercise-translation').style.display = 'none';
    document.getElementById('exercise-options').style.display = 'none';
    document.getElementById('exercise-feedback').classList.add('hidden');
    document.getElementById('next-exercise-btn').classList.add('hidden');
    document.getElementById('mark-learned-btn').classList.add('hidden');

    // Create matching container
    const contentDiv = document.getElementById('exercise-content');
    let matchingHTML = `
        <div class="matching-progress">
            Matched: <span id="matched-count">0</span> / ${pairs.length}
        </div>
        <div class="matching-container">
            <div class="matching-column" id="left-column">
                ${leftItems.map(item => `
                    <div class="matching-card" data-id="${item.id}" data-side="left">
                        ${item.text}
                    </div>
                `).join('')}
            </div>
            <div class="matching-column" id="right-column">
                ${rightItems.map(item => `
                    <div class="matching-card" data-id="${item.id}" data-side="right">
                        ${item.text}
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

function renderExercise() {
    const ex = state.currentExercise;

    // Make sure content is visible and loading is hidden
    document.getElementById('exercise-loading').classList.add('hidden');
    document.getElementById('exercise-content').style.display = 'block';

    // Update counter
    document.getElementById('exercise-counter').textContent = `Exercise ${state.exerciseCount}`;

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

    document.getElementById('exercise-question').innerHTML = `
        ${questionText}<br><strong>"${targetWord}"</strong>
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
    const words = ex.sentence.split(/\s+/); // Split by whitespace
    clickableContainer.innerHTML = words.map((word, index) => {
        // Clean up punctuation for comparison but keep it for display
        // Remove common punctuation marks (must match backend normalization)
        const cleanWord = word.replace(/[.,!?;:»«"""'()—–\-\[\]]/g, '');
        return `<span class="sentence-word" data-word="${cleanWord}" data-index="${index}">${word}</span>`;
    }).join('');

    // Add click handlers to words
    document.querySelectorAll('.sentence-word').forEach(wordEl => {
        wordEl.addEventListener('click', () => handleWordClick(wordEl));
    });

    document.getElementById('exercise-translation').textContent = `Translation: ${ex.sentence_translation}`;

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
    // Check if user has at least 10 words for matching exercise
    const hasEnoughWords = state.learningWords.length >= 10;

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

    // Word type filter dropdown
    const wordTypeSelect = document.getElementById('word-type-select');
    if (wordTypeSelect) {
        wordTypeSelect.addEventListener('change', () => {
            updateSelectedWordType();
        });
    }

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
