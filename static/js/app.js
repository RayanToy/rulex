// State
let currentUser = null;
let testQuestions = [];
let currentQuestion = 0;
let selectedAnswer = null;
let testAnswers = [];
let selectedGrade = 6; // Класс ученика
let selectedWordClass = 5; // Класс слов (на 1 меньше)
// Сложность и подсчёт верных ответов живут на сервере:
// браузер больше не знает правильных ответов.
let questionsData = [];
// Текст из API и полей ввода вставляется в разметку только через escapeHtml:
// толкования и дистракторы пишет модель или преподаватель, и без экранирования
// текст вида <img src=x onerror=…> исполнялся бы в браузере каждого ученика.
function escapeHtml(value) {
    const entities = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
    return String(value ?? '').replace(/[&<>"']/g, ch => entities[ch]);
}
// Функция для получения названия класса
function getGradeLabel(grade) {
    if (grade === 12) return 'Выпускник';
    return grade + ' класс';
}
// Генерация кнопок выбора класса
function generateGradeButtons() {
    const container = document.getElementById('grade-selector');
    if (!container) return;

    let html = '';
    for (let g = 3; g <= 12; g++) {
        const label = g === 12 ? 'Вып.' : g;
        const sublabel = g === 12 ? '' : 'класс';
        const selected = g === selectedGrade ? 'selected' : '';
        html += `<button class="grade-btn ${selected}" onclick="selectGrade(${g}, this)">${label}<span>${sublabel}</span></button>`;
    }
    container.innerHTML = html;
}
// ============ AUTH ============
function showLogin() {
    document.getElementById('login-form').classList.remove('hidden');
    document.getElementById('register-form').classList.add('hidden');
}
function showRegister() {
    document.getElementById('login-form').classList.add('hidden');
    document.getElementById('register-form').classList.remove('hidden');
}
async function handleLogin() {
    const username = document.getElementById('login-username').value.trim();
    const password = document.getElementById('login-password').value;

    if (!username || !password) {
        showError('login-error', 'Заполните все поля');
        return;
    }

    try {
        const response = await fetch('/api/auth/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password })
        });

        const data = await response.json();

        if (response.ok) {
            currentUser = data.user;
            showApp();
        } else {
            showError('login-error', data.detail || 'Ошибка входа');
        }
    } catch (e) {
        showError('login-error', 'Ошибка соединения');
    }
}
async function handleRegister() {
    const username = document.getElementById('register-username').value.trim();
    const email = document.getElementById('register-email').value.trim();
    const fullName = document.getElementById('register-fullname').value.trim();
    const grade = parseInt(document.getElementById('register-grade').value);
    const password = document.getElementById('register-password').value;

    if (!username || !email || !password) {
        showError('register-error', 'Заполните обязательные поля');
        return;
    }

    try {
        const response = await fetch('/api/auth/register', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, email, password, full_name: fullName, grade })
        });

        const data = await response.json();

        if (response.ok) {
            currentUser = data.user;
            showApp();
        } else {
            showError('register-error', data.detail || 'Ошибка регистрации');
        }
    } catch (e) {
        showError('register-error', 'Ошибка соединения');
    }
}
async function handleLogout() {
    try {
        await fetch('/api/auth/logout', { method: 'POST' });
    } catch (e) {}

    currentUser = null;
    document.getElementById('auth-container').classList.remove('hidden');
    document.getElementById('app-container').classList.add('hidden');
    updateHeaderAuth();
}
async function checkAuth() {
    try {
        const response = await fetch('/api/auth/me');
        if (response.ok) {
            currentUser = await response.json();
            showApp();
        }
    } catch (e) {}
}
function showApp() {
    document.getElementById('auth-container').classList.add('hidden');
    document.getElementById('app-container').classList.remove('hidden');
    updateHeaderAuth();
    selectedGrade = currentUser.grade || 6;
    selectedWordClass = selectedGrade - 1;
    generateGradeButtons();
}
function updateHeaderAuth() {
    const headerAuth = document.getElementById('header-auth');
    if (currentUser) {
        headerAuth.innerHTML = `
            <div class="user-info">
                <div>
                    <span class="user-name">${escapeHtml(currentUser.full_name || currentUser.username)}</span>
                    <span class="user-grade">${getGradeLabel(currentUser.grade || 6)}</span>
                </div>
                <button class="btn btn-outline btn-sm" onclick="handleLogout()">Выйти</button>
            </div>
        `;
    } else {
        headerAuth.innerHTML = '';
    }
}
function showError(elementId, message) {
    const el = document.getElementById(elementId);
    el.textContent = message;
    el.classList.remove('hidden');
    setTimeout(() => el.classList.add('hidden'), 5000);
}
// ============ NAVIGATION ============
function showPage(pageName, tab) {
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));

    document.getElementById('page-' + pageName).classList.add('active');
    if (tab) tab.classList.add('active');
    if (pageName === 'database') loadQuestions();
    if (pageName === 'test') { generateGradeButtons(); loadTestStats(); }
    if (pageName === 'history') loadHistory();
}
// ============ ADAPTIVE TEST ============
function selectGrade(grade, btn) {
    selectedGrade = grade;
    selectedWordClass = grade - 1; // Класс слов на 1 меньше
    document.querySelectorAll('.grade-btn').forEach(b => b.classList.remove('selected'));
    btn.classList.add('selected');
    loadTestStats();
}
async function loadTestStats() {
    try {
        const response = await fetch('/api/questions');
        const questions = await response.json();
        // Фильтруем по классу слов (word_class)
        const gradeQuestions = questions.filter(q => q.word_class === selectedWordClass);
        document.getElementById('available-questions').textContent = gradeQuestions.length;

        const btn = document.getElementById('btn-start-test');
        const autoGenBtn = document.getElementById('btn-auto-gen-test');

        if (gradeQuestions.length < 5) {
            btn.disabled = true;
            btn.textContent = 'Недостаточно вопросов (мин. 5)';
            if (autoGenBtn) autoGenBtn.classList.remove('hidden');
        } else {
            btn.disabled = false;
            btn.textContent = 'Начать тест';
            if (autoGenBtn) autoGenBtn.classList.add('hidden');
        }
    } catch (e) {}
}
async function startAdaptiveTest() {
    try {
        const response = await fetch('/api/test/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ grade: selectedWordClass }) // Передаём класс слов
        });

        const data = await response.json();

        if (!response.ok) {
            alert(data.detail || 'Ошибка запуска теста');
            return;
        }
        testQuestions = data.questions;
        currentQuestion = 0;
        selectedAnswer = null;
        testAnswers = [];
        document.getElementById('test-setup').classList.add('hidden');
        document.getElementById('test-result').classList.add('hidden');
        document.getElementById('test-question').classList.remove('hidden');
        document.getElementById('total-num').textContent = testQuestions.length;

        showQuestion();
    } catch (e) {
        alert('Ошибка загрузки теста');
    }
}
function updateDifficultyDots(difficulty) {
    // Показываем сложность текущего вопроса (её сервер присылает
    // и она не выдаёт ответ). Раньше здесь рисовался клиентский
    // счётчик верных ответов — вместе с ним ушёл и подсчёт.
    const level = difficulty || 5;
    const dots = document.querySelectorAll('.difficulty-dot');
    dots.forEach((dot, index) => {
        dot.classList.remove('active', 'high');
        if (index < level) {
            dot.classList.add('active');
            if (level >= 8) {
                dot.classList.add('high');
            }
        }
    });
}
function showQuestion() {
    const q = testQuestions[currentQuestion];

    document.getElementById('question-number').textContent = `Вопрос ${currentQuestion + 1}`;
    document.getElementById('question-text').textContent = q.question;
    document.getElementById('current-num').textContent = currentQuestion + 1;

    const progress = (currentQuestion / testQuestions.length) * 100;
    document.getElementById('progress-bar').style.width = progress + '%';
    document.getElementById('progress-percent').textContent = Math.round(progress);
    updateDifficultyDots(q.difficulty);
    const container = document.getElementById('options-container');
    container.innerHTML = '';

    q.options.forEach((option, index) => {
        const div = document.createElement('div');
        div.className = 'option-item';
        div.innerHTML = `
            <div class="option-radio"></div>
            <span class="option-text">${escapeHtml(option)}</span>
        `;
        div.onclick = () => selectOption(index, div, option);
        container.appendChild(div);
    });
    selectedAnswer = null;
    document.getElementById('next-btn').disabled = true;
    document.getElementById('next-btn').textContent =
        currentQuestion === testQuestions.length - 1 ? 'Завершить тест' : 'Следующий вопрос';
}
function selectOption(index, element, answer) {
    document.querySelectorAll('.option-item').forEach(el => el.classList.remove('selected'));
    element.classList.add('selected');
    selectedAnswer = { index, answer };
    document.getElementById('next-btn').disabled = false;
}
function nextQuestion() {
    if (selectedAnswer === null) return;

    const q = testQuestions[currentQuestion];

    // Правильный ответ браузеру больше не известен — проверка
    // и подсчёт сложности делаются на сервере. Отправляем только
    // то, что выбрал ученик.
    testAnswers.push({
        question_id: q.id,
        answer: selectedAnswer.answer
    });

    currentQuestion++;

    if (currentQuestion >= testQuestions.length) {
        completeTest();
    } else {
        showQuestion();
    }
}
async function completeTest() {
    try {
        const response = await fetch('/api/test/complete', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                answers: testAnswers,
                grade: selectedWordClass
            })
        });

        const result = await response.json();

        document.getElementById('test-question').classList.add('hidden');
        document.getElementById('test-result').classList.remove('hidden');

        const resultIcon = document.getElementById('result-icon');
        resultIcon.className = 'result-icon ' + result.level;

        const levelText = document.getElementById('result-level-text');
        levelText.textContent = `Уровень: ${result.level_text}`;
        levelText.className = 'result-level ' + result.level;

        document.getElementById('result-grade-text').textContent =
            `Тестирование для ${getGradeLabel(selectedGrade)}`;

        document.getElementById('score-value').textContent =
            `${result.score}/${result.total}`;
        document.getElementById('score-text').textContent =
            `${result.percentage}% правильных ответов`;

        document.getElementById('high-freq-result').textContent =
            `${result.details.high_freq.correct}/${result.details.high_freq.total}`;
        document.getElementById('medium-freq-result').textContent =
            `${result.details.medium_freq.correct}/${result.details.medium_freq.total}`;
        document.getElementById('low-freq-result').textContent =
            `${result.details.low_freq.correct}/${result.details.low_freq.total}`;

        document.getElementById('recommendation-text').textContent = result.recommendation;

    } catch (e) {
        alert('Ошибка сохранения результатов');
    }
}
function restartTest() {
    document.getElementById('test-result').classList.add('hidden');
    document.getElementById('test-setup').classList.remove('hidden');
    generateGradeButtons();
    loadTestStats();
}
// ============ HISTORY ============
async function loadHistory() {
    try {
        const response = await fetch('/api/test/history');
        const history = await response.json();

        const container = document.getElementById('history-container');

        if (history.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <p>Вы ещё не проходили тестирование</p>
                    <button class="btn btn-primary" style="margin-top: 16px;"
                        onclick="showPage('test', document.querySelectorAll('.nav-tab')[1])">
                        Пройти тест
                    </button>
                </div>
            `;
            return;
        }

        container.innerHTML = history.map(h => `
            <div class="history-item">
                <div class="history-info">
                    <div class="history-score">${h.percentage}%</div>
                    <div>
                        <div class="history-details">
                            <span class="badge badge-${h.level === 'high' ? 'green' : h.level === 'medium' ? 'yellow' : 'red'}">
                                ${h.level === 'high' ? 'Высокий' : h.level === 'medium' ? 'Средний' : 'Низкий'}
                            </span>
                            ${getGradeLabel(h.grade + 1)} • ${h.score}/${h.total} ответов
                        </div>
                    </div>
                </div>
                <div class="history-date">${new Date(h.completed_at).toLocaleDateString('ru-RU')}</div>
            </div>
        `).join('');
    } catch (e) {}
}
// ============ GENERATION ============
async function generateQuestion() {
    const word = document.getElementById('word-input').value.trim();
    if (!word) { alert('Введите слово'); return; }
    const btn = document.getElementById('btn-generate');
    btn.disabled = true;
    btn.innerHTML = '<span class="loading"></span>Генерация...';
    try {
        const response = await fetch('/api/generate-and-save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ word })
        });
        const result = await response.json();
        if (response.ok) {
            document.getElementById('generate-result').classList.remove('hidden');
            document.getElementById('generate-result').innerHTML = `
                <div class="generated-result">
                    <h4>✅ Вопрос сгенерирован!</h4>
                    <div class="result-item"><span class="result-label">Толкование:</span> ${escapeHtml(result.question.question)}</div>
                    <div class="result-item"><span class="result-label">Ответ:</span> ${escapeHtml(result.question.target_word)}</div>
                </div>
            `;
        } else {
            throw new Error(result.detail || 'Ошибка генерации');
        }
    } catch (e) {
        document.getElementById('generate-result').classList.remove('hidden');
        document.getElementById('generate-result').innerHTML = `<div class="alert alert-error">Ошибка: ${escapeHtml(e.message)}</div>`;
    }
    btn.disabled = false;
    btn.textContent = 'Сгенерировать вопрос';
}
async function generateMultiple() {
    const words = document.getElementById('words-input').value.split('\n').map(w => w.trim()).filter(w => w);
    if (words.length === 0) { alert('Введите слова'); return; }
    const btn = document.getElementById('btn-generate-multiple');
    btn.disabled = true;
    btn.innerHTML = '<span class="loading"></span>Генерация...';
    const resultDiv = document.getElementById('generate-multiple-result');
    resultDiv.classList.remove('hidden');
    resultDiv.innerHTML = '<div class="alert alert-info">Генерация вопросов...</div>';
    let successCount = 0;
    let results = [];
    for (const word of words) {
        try {
            const response = await fetch('/api/generate-and-save', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ word })
            });
            if (response.ok) {
                successCount++;
                results.push(`✅ ${escapeHtml(word)}`);
            } else {
                results.push(`❌ ${escapeHtml(word)}`);
            }
        } catch (e) {
            results.push(`❌ ${escapeHtml(word)}`);
        }
    }
    resultDiv.innerHTML = `
        <div class="generated-result">
            <h4>Результат: ${successCount}/${words.length}</h4>
            <div>${results.join('<br>')}</div>
        </div>
    `;
    btn.disabled = false;
    btn.textContent = 'Сгенерировать все';
}
async function autoGenerateForTest() {
    const btn = document.getElementById('btn-auto-gen-test');
    btn.disabled = true;
    btn.innerHTML = '<span class="loading"></span>Генерация... (1-2 минуты)';

    try {
        const response = await fetch('/api/auto-generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ grade: selectedWordClass })
        });

        const data = await response.json();

        if (response.ok) {
            alert('✅ ' + data.message);
            loadTestStats();
        } else {
            throw new Error(data.detail || 'Ошибка генерации');
        }
    } catch (e) {
        alert('❌ Ошибка: ' + e.message);
    }

    btn.disabled = false;
    btn.innerHTML = '🔄 Автоматически сгенерировать вопросы для этого класса';
}
async function autoGenerate() {
    const wordClass = parseInt(document.getElementById('auto-class-select').value);
    const btn = document.getElementById('btn-auto-generate');
    const resultDiv = document.getElementById('auto-generate-result');

    btn.disabled = true;
    btn.innerHTML = '<span class="loading"></span>Генерация... (1-2 минуты)';

    resultDiv.classList.remove('hidden');
    resultDiv.innerHTML = '<div class="alert alert-info">⏳ Генерация вопросов из списка слов...</div>';

    try {
        const response = await fetch('/api/auto-generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ grade: wordClass })
        });

        const data = await response.json();

        if (response.ok) {
            resultDiv.innerHTML = `
                <div class="generated-result">
                    <h4>✅ ${escapeHtml(data.message)}</h4>
                </div>
            `;
        } else {
            throw new Error(data.detail || 'Ошибка генерации');
        }
    } catch (e) {
        resultDiv.innerHTML = `<div class="alert alert-error">❌ Ошибка: ${escapeHtml(e.message)}</div>`;
    }

    btn.disabled = false;
    btn.innerHTML = '🔄 Сгенерировать 20 вопросов';
}
async function saveManualQuestion() {
    const definition = document.getElementById('manual-definition').value.trim();
    const correct = document.getElementById('manual-correct').value.trim();

    if (!definition || !correct) {
        alert('Заполните толкование и правильный ответ');
        return;
    }
    try {
        const response = await fetch('/api/questions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                target_word: correct,
                definition: definition,
                correct_answer: correct,
                distractor_1: document.getElementById('manual-d1').value.trim() || null,
                distractor_2: document.getElementById('manual-d2').value.trim() || null,
                distractor_3: document.getElementById('manual-d3').value.trim() || null,
                word_class: parseInt(document.getElementById('manual-class').value),
                frequency_type: document.getElementById('manual-frequency').value,
                difficulty: parseInt(document.getElementById('manual-difficulty').value)
            })
        });
        if (response.ok) {
            document.getElementById('manual-result').classList.remove('hidden');
            document.getElementById('manual-result').innerHTML =
                `<div class="alert alert-success" style="margin-top: 16px;">✅ Вопрос сохранён!</div>`;

            // Очистка полей
            document.getElementById('manual-definition').value = '';
            document.getElementById('manual-correct').value = '';
            document.getElementById('manual-d1').value = '';
            document.getElementById('manual-d2').value = '';
            document.getElementById('manual-d3').value = '';
        } else {
            throw new Error('Ошибка сохранения');
        }
    } catch (e) {
        document.getElementById('manual-result').classList.remove('hidden');
        document.getElementById('manual-result').innerHTML =
            `<div class="alert alert-error" style="margin-top: 16px;">Ошибка: ${escapeHtml(e.message)}</div>`;
    }
}
// ============ DATABASE ============
async function loadQuestions() {
    try {
        const response = await fetch('/api/questions/full');
        const questions = await response.json();
        questionsData = questions;
        const container = document.getElementById('questions-container');
        if (questions.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <p>В базе пока нет вопросов</p>
                    <button class="btn btn-primary" style="margin-top: 16px;"
                        onclick="showPage('generate', document.querySelectorAll('.nav-tab')[3])">
                        Добавить вопросы
                    </button>
                </div>
            `;
            return;
        }
        const freqBadge = (type) => {
            const badges = { 'high': 'badge-green', 'medium': 'badge-yellow', 'low': 'badge-red' };
            const labels = { 'high': 'Выс.', 'medium': 'Ср.', 'low': 'Низ.' };
            return `<span class="badge ${badges[type] || 'badge-yellow'}">${labels[type] || 'Ср.'}</span>`;
        };
        container.innerHTML = `
            <table class="questions-table">
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Слово</th>
                        <th>Толкование</th>
                        <th>Кл.</th>
                        <th>Част.</th>
                        <th>Действия</th>
                    </tr>
                </thead>
                <tbody id="questions-tbody"></tbody>
            </table>
        `;
        const tbody = document.getElementById('questions-tbody');
        questions.forEach(q => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${q.id}</td>
                <td><span class="badge badge-blue">${escapeHtml(q.target_word)}</span></td>
                <td class="tooltip-container">
                    <span class="cell-definition">${escapeHtml(q.question.substring(0, 30))}...</span>
                    <div class="tooltip-text">${escapeHtml(q.question)}</div>
                </td>
                <td>${q.word_class}</td>
                <td>${freqBadge(q.frequency_type)}</td>
                <td>
                    <button class="btn btn-edit" onclick="openEditModal(${q.id})">Ред.</button>
                    <button class="btn btn-danger btn-sm" onclick="deleteQuestion(${q.id})">Удалить</button>
                </td>
            `;
            tbody.appendChild(tr);
        });
    } catch (e) {
        console.error('Error loading questions:', e);
    }
}
function openEditModal(id) {
    const q = questionsData.find(item => item.id === id);
    if (!q) return;

    document.getElementById('edit-id').value = q.id;
    document.getElementById('edit-word').value = q.target_word;
    document.getElementById('edit-definition').value = q.question;
    document.getElementById('edit-d1').value = q.distractor_1 || '';
    document.getElementById('edit-d2').value = q.distractor_2 || '';
    document.getElementById('edit-d3').value = q.distractor_3 || '';
    document.getElementById('edit-class').value = q.word_class || 6;
    document.getElementById('edit-frequency').value = q.frequency_type || 'medium';
    document.getElementById('edit-difficulty').value = q.difficulty || 5;

    document.getElementById('edit-modal').classList.add('active');
}
function closeEditModal() {
    document.getElementById('edit-modal').classList.remove('active');
}
async function saveEditedQuestion() {
    const id = document.getElementById('edit-id').value;
    const data = {
        target_word: document.getElementById('edit-word').value,
        definition: document.getElementById('edit-definition').value,
        correct_answer: document.getElementById('edit-word').value,
        distractor_1: document.getElementById('edit-d1').value || null,
        distractor_2: document.getElementById('edit-d2').value || null,
        distractor_3: document.getElementById('edit-d3').value || null,
        word_class: parseInt(document.getElementById('edit-class').value),
        frequency_type: document.getElementById('edit-frequency').value,
        difficulty: parseInt(document.getElementById('edit-difficulty').value)
    };

    try {
        const response = await fetch(`/api/questions/${id}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        if (response.ok) {
            closeEditModal();
            loadQuestions();
        } else {
            alert('Ошибка сохранения');
        }
    } catch (e) {
        alert('Ошибка: ' + e.message);
    }
}
async function deleteQuestion(id) {
    if (!confirm('Удалить вопрос?')) return;
    try {
        const response = await fetch(`/api/questions/${id}`, { method: 'DELETE' });
        if (response.ok) {
            loadQuestions();
        }
    } catch (e) {
        alert('Ошибка удаления');
    }
}
// ============ INIT ============
document.addEventListener('DOMContentLoaded', () => {
    checkAuth();
});
