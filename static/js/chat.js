// Global variables
let currentLanguage = 'fr'; // Default language
let conversation = [];

// DOM Elements
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const messageContainer = document.getElementById('messageContainer');
const clearBtn = document.getElementById('clearBtn');
const frBtn = document.getElementById('frBtn');
const arBtn = document.getElementById('arBtn');
const loadingIndicator = document.getElementById('loadingIndicator');
const exampleItems = document.querySelectorAll('.example-item');

// UI Text elements
const infoTitle = document.getElementById('infoTitle');
const infoText = document.getElementById('infoText');
const examplesTitle = document.getElementById('examplesTitle');
const welcomeMessage = document.getElementById('welcomeMessage');
const chatTitle = document.getElementById('chatTitle');
const clearText = document.getElementById('clearText');

// Text translations
const translations = {
    fr: {
        infoTitle: "Informations RNE",
        infoText: "Cet assistant vous aide à naviguer les lois et procédures du Registre National des Entreprises en Tunisie.",
        examplesTitle: "Exemples de questions",
        welcomeMessage: "Bonjour! Je suis l'assistant RNE, spécialisé dans les lois et procédures du Registre National des Entreprises en Tunisie. Comment puis-je vous aider aujourd'hui?",
        chatTitle: "Assistant RNE",
        clearText: "Effacer la conversation",
        inputPlaceholder: "Tapez votre question ici...",
        examples: [
            "Quels sont les documents nécessaires pour l'immatriculation d'une SARL?",
            "Quel est le délai pour déposer les états financiers?",
            "Quelles sont les redevances pour la modification du capital social?"
        ]
    },
    ar: {
        infoTitle: "معلومات السجل الوطني للمؤسسات",
        infoText: "يساعدك هذا المساعد على التنقل في قوانين وإجراءات السجل الوطني للمؤسسات في تونس.",
        examplesTitle: "أمثلة على الأسئلة",
        welcomeMessage: "مرحبًا! أنا مساعد السجل الوطني للمؤسسات، متخصص في قوانين وإجراءات السجل في تونس. كيف يمكنني مساعدتك اليوم؟",
        chatTitle: "مساعد السجل الوطني للمؤسسات",
        clearText: "مسح المحادثة",
        inputPlaceholder: "اكتب سؤالك هنا...",
        examples: [
            "ما هي الوثائق اللازمة لتسجيل شركة ذات مسؤولية محدودة؟",
            "ما هي مهلة إيداع القوائم المالية؟",
            "ما هي الرسوم المطلوبة لتعديل رأس المال؟"
        ]
    }
};

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    // Set initial translations
    updateUILanguage(currentLanguage);
    
    // Event listeners
    sendBtn.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    clearBtn.addEventListener('click', clearConversation);
    
    // Language toggle
    frBtn.addEventListener('click', function() {
        setLanguage('fr');
    });
    
    arBtn.addEventListener('click', function() {
        setLanguage('ar');
    });
    
    // Example questions
    exampleItems.forEach(item => {
        item.addEventListener('click', function() {
            userInput.value = this.textContent;
            sendMessage();
        });
    });
});

// Send message to the backend
function sendMessage() {
    const message = userInput.value.trim();
    
    if (!message) {
        return;
    }
    
    // Add user message to UI
    addMessageToUI('user', message);
    
    // Clear input
    userInput.value = '';
    
    // Show loading indicator
    loadingIndicator.style.display = 'flex';
    
    // Send to backend
    fetch('/api/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            message: message,
            language: currentLanguage
        })
    })
    .then(response => response.json())
    .then(data => {
        // Hide loading indicator
        loadingIndicator.style.display = 'none';
        
        // Add bot response to UI
        addMessageToUI('bot', data.response, data.references);
        
        // If the response has a different language, update the UI
        if (data.language && data.language !== currentLanguage) {
            setLanguage(data.language, false);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        loadingIndicator.style.display = 'none';
        
        // Add error message
        const errorMessage = currentLanguage === 'fr' 
            ? "Désolé, une erreur s'est produite. Veuillez réessayer."
            : "عذرًا، حدث خطأ. يرجى المحاولة مرة أخرى.";
            
        addMessageToUI('bot', errorMessage);
    });
}

// Add message to the UI
function addMessageToUI(sender, message, references = []) {
    // Create message container
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    
    // Create message content
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    // Format message with Markdown
    contentDiv.innerHTML = marked.parse(message);
    
    // Add references if any
    if (references && references.length > 0) {
        const referencesDiv = document.createElement('div');
        referencesDiv.className = 'references';
        
        references.forEach(ref => {
            if (ref.pdf_link) {
                const refLink = document.createElement('a');
                refLink.href = ref.pdf_link;
                refLink.className = 'reference-link';
                refLink.target = '_blank';
                refLink.textContent = currentLanguage === 'fr' 
                    ? `📄 Document ${ref.code}` 
                    : `📄 الوثيقة ${ref.code}`;
                referencesDiv.appendChild(refLink);
                referencesDiv.appendChild(document.createElement('br'));
            }
        });
        
        contentDiv.appendChild(referencesDiv);
    }
    
    // Add timestamp
    const timestamp = document.createElement('div');
    timestamp.className = 'message-timestamp';
    
    const now = new Date();
    const timeString = now.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
    timestamp.textContent = timeString;
    
    // Append elements
    messageDiv.appendChild(contentDiv);
    messageDiv.appendChild(timestamp);
    messageContainer.appendChild(messageDiv);
    
    // Save to conversation history
    conversation.push({
        sender: sender,
        message: message,
        timestamp: now.toISOString(),
        references: references
    });
    
    // Scroll to bottom
    messageContainer.scrollTop = messageContainer.scrollHeight;
}

// Clear the conversation
function clearConversation() {
    // Remove all messages except the welcome message
    while (messageContainer.children.length > 1) {
        messageContainer.removeChild(messageContainer.lastChild);
    }
    
    // Reset conversation array
    conversation = [];
}

// Set the language
function setLanguage(lang, updateWelcome = true) {
    currentLanguage = lang;
    
    // Update UI
    if (lang === 'fr') {
        frBtn.classList.add('active');
        arBtn.classList.remove('active');
        document.documentElement.setAttribute('dir', 'ltr');
    } else {
        frBtn.classList.remove('active');
        arBtn.classList.add('active');
        document.documentElement.setAttribute('dir', 'rtl');
    }
    
    // Update text
    updateUILanguage(lang, updateWelcome);
}

// Update UI text based on language
function updateUILanguage(lang, updateWelcome = true) {
    const texts = translations[lang];
    
    infoTitle.textContent = texts.infoTitle;
    infoText.textContent = texts.infoText;
    examplesTitle.textContent = texts.examplesTitle;
    chatTitle.textContent = texts.chatTitle;
    clearText.textContent = texts.clearText;
    userInput.placeholder = texts.inputPlaceholder;
    
    // Update welcome message if needed
    if (updateWelcome) {
        welcomeMessage.textContent = texts.welcomeMessage;
    }
    
    // Update example questions
    const examplesList = document.querySelectorAll('.example-item');
    examplesList.forEach((item, index) => {
        if (texts.examples[index]) {
            item.textContent = texts.examples[index];
        }
    });
}