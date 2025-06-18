// Enhanced Frontend JavaScript - Elite Agent Command Center Interface

// ============================================================================
// STATE MANAGEMENT & CONFIGURATION
// ============================================================================

class EliteAgentInterface {
    constructor() {
        this.sessionId = this.getSessionId();
        this.userContext = this.loadUserContext();
        this.attachedFiles = [];
        this.currentWorkflow = null;
        this.agentStatuses = {};
        this.systemMetrics = {
            totalRequests: 0,
            successRate: 1.0,
            avgResponseTime: 0.0
        };
        
        this.initializeInterface();
        this.loadAgentStatuses();
        this.startPerformanceMonitoring();
    }
    
    getSessionId() {
        let sessionId = localStorage.getItem('eliteAgentSessionId');
        if (!sessionId) {
            sessionId = `elite-${Date.now()}-${Math.random().toString(36).substring(2, 11)}`;
            localStorage.setItem('eliteAgentSessionId', sessionId);
        }
        return sessionId;
    }
    
    loadUserContext() {
        const saved = localStorage.getItem('eliteAgentContext');
        return saved ? JSON.parse(saved) : {
            business_type: '',
            industry: '',
            target_audience: '',
            current_revenue: '',
            main_challenges: [],
            goals: []
        };
    }
    
    saveUserContext() {
        localStorage.setItem('eliteAgentContext', JSON.stringify(this.userContext));
    }
    
    initializeInterface() {
        this.dom = {
            // Core elements
            messageInput: document.getElementById('messageInput'),
            sendButton: document.getElementById('sendButton'),
            chatMessages: document.getElementById('chatMessages'),
            
            // Context form
            businessType: document.getElementById('businessType'),
            industry: document.getElementById('industry'),
            revenue: document.getElementById('revenue'),
            audience: document.getElementById('audience'),
            challenges: document.getElementById('challenges'),
            goals: document.getElementById('goals'),
            saveContextBtn: document.getElementById('saveContextBtn'),
            
            // Agent mode selection
            autoMode: document.getElementById('autoMode'),
            eliteMode: document.getElementById('eliteMode'),
            standardMode: document.getElementById('standardMode'),
            eliteSelector: document.getElementById('eliteSelector'),
            specificAgent: document.getElementById('specificAgent'),
            
            // File handling
            dropZone: document.getElementById('dropZone'),
            fileInput: document.getElementById('fileInput'),
            fileAttachments: document.getElementById('fileAttachments'),
            
            // Status and metrics
            agentStatusBar: document.getElementById('agentStatusBar'),
            sessionStatus: document.getElementById('session-status'),
            modelStatus: document.getElementById('model-status'),
            costStatus: document.getElementById('cost-status'),
            timeStatus: document.getElementById('time-status'),
            agentsStatus: document.getElementById('agents-status'),
            
            // Workflow and performance
            workflowPanel: document.getElementById('workflowPanel'),
            workflowSteps: document.getElementById('workflowSteps'),
            performanceStats: document.getElementById('performanceStats'),
            totalRequests: document.getElementById('totalRequests'),
            successRate: document.getElementById('successRate'),
            avgResponseTime: document.getElementById('avgResponseTime'),
            activityLog: document.getElementById('activityLog')
        };
        
        this.setupEventListeners();
        this.populateContextForm();
        this.updateSessionStatus();
    }
    
    setupEventListeners() {
        // Core chat functionality
        this.dom.sendButton.addEventListener('click', () => this.handleSendMessage());
        this.dom.messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.handleSendMessage();
            }
        });
        
        // Context management
        this.dom.saveContextBtn.addEventListener('click', () => this.saveBusinessContext());
        
        // Auto-save context on input changes
        [this.dom.businessType, this.dom.industry, this.dom.revenue, this.dom.audience].forEach(input => {
            input.addEventListener('change', () => this.autoSaveContext());
        });
        
        [this.dom.challenges, this.dom.goals].forEach(textarea => {
            textarea.addEventListener('blur', () => this.autoSaveContext());
        });
        
        // Agent mode selection
        document.querySelectorAll('input[name="agentMode"]').forEach(radio => {
            radio.addEventListener('change', () => this.handleModeChange());
        });
        
        // File handling
        this.setupFileHandling();
        
        // Auto-resize message input
        this.dom.messageInput.addEventListener('input', () => this.autoResizeInput());
    }
    
    setupFileHandling() {
        this.dom.dropZone.addEventListener('click', () => this.dom.fileInput.click());
        
        this.dom.dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.dom.dropZone.classList.add('dragover');
        });
        
        this.dom.dropZone.addEventListener('dragleave', () => {
            this.dom.dropZone.classList.remove('dragover');
        });
        
        this.dom.dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            this.dom.dropZone.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length) this.handleFiles(files);
        });
        
        this.dom.fileInput.addEventListener('change', () => {
            if (this.dom.fileInput.files.length) {
                this.handleFiles(this.dom.fileInput.files);
            }
        });
    }
    
    // ============================================================================
    // AGENT STATUS & PERFORMANCE MONITORING
    // ============================================================================
    
    async loadAgentStatuses() {
        try {
            const response = await fetch('/elite/agents/status');
            if (response.ok) {
                this.agentStatuses = await response.json();
                this.updateAgentStatusBar();
            }
        } catch (error) {
            console.error('Failed to load agent statuses:', error);
            this.updateAgentStatusBar(true);
        }
    }
    
    updateAgentStatusBar(isError = false) {
        if (isError) {
            this.dom.agentStatusBar.innerHTML = '<div class="agent-status error">❌ Elite agents offline</div>';
            return;
        }
        
        const statusHTML = Object.entries(this.agentStatuses).map(([agentName, status]) => {
            const statusClass = status.status === 'idle' ? 'online' : 
                               status.status === 'busy' ? 'busy' : 'error';
            const emoji = status.status === 'idle' ? '🟢' : 
                         status.status === 'busy' ? '🟡' : '🔴';
            
            return `<div class="agent-status ${statusClass}">${emoji} ${agentName}</div>`;
        }).join('');
        
        this.dom.agentStatusBar.innerHTML = statusHTML || '<div class="agent-status loading">Loading agents...</div>';
    }
    
    async loadSystemMetrics() {
        try {
            const response = await fetch('/elite/metrics');
            if (response.ok) {
                const metrics = await response.json();
                this.updatePerformanceDisplay(metrics);
            }
        } catch (error) {
            console.error('Failed to load system metrics:', error);
        }
    }
    
    updatePerformanceDisplay(metrics) {
        if (metrics.performance) {
            this.dom.totalRequests.textContent = metrics.performance.total_requests || 0;
            this.dom.successRate.textContent = `${Math.round((metrics.performance.successful_requests / Math.max(1, metrics.performance.total_requests)) * 100)}%`;
            this.dom.avgResponseTime.textContent = `${metrics.performance.average_response_time?.toFixed(1) || 0.0}s`;
        }
    }
    
    startPerformanceMonitoring() {
        // Update agent statuses every 30 seconds
        setInterval(() => {
            this.loadAgentStatuses();
        }, 30000);
        
        // Update system metrics every 60 seconds
        setInterval(() => {
            this.loadSystemMetrics();
        }, 60000);
    }
    
    // ============================================================================
    // CONTEXT MANAGEMENT
    // ============================================================================
    
    populateContextForm() {
        this.dom.businessType.value = this.userContext.business_type || '';
        this.dom.industry.value = this.userContext.industry || '';
        this.dom.revenue.value = this.userContext.current_revenue || '';
        this.dom.audience.value = this.userContext.target_audience || '';
        this.dom.challenges.value = Array.isArray(this.userContext.main_challenges) ? 
            this.userContext.main_challenges.join(', ') : (this.userContext.main_challenges || '');
        this.dom.goals.value = Array.isArray(this.userContext.goals) ? 
            this.userContext.goals.join(', ') : (this.userContext.goals || '');
    }
    
    autoSaveContext() {
        this.userContext = {
            business_type: this.dom.businessType.value,
            industry: this.dom.industry.value,
            current_revenue: this.dom.revenue.value,
            target_audience: this.dom.audience.value,
            main_challenges: this.dom.challenges.value.split(',').map(s => s.trim()).filter(s => s),
            goals: this.dom.goals.value.split(',').map(s => s.trim()).filter(s => s)
        };
        this.saveUserContext();
    }
    
    saveBusinessContext() {
        this.autoSaveContext();
        this.showNotification('✅ Business context saved', 'success');
    }
    
    // ============================================================================
    // MESSAGE HANDLING & AGENT COORDINATION
    // ============================================================================
    
    async handleSendMessage() {
        const message = this.dom.messageInput.value.trim();
        if (!message) return;
        
        // Add user message to chat
        this.addMessage(message, 'user');
        this.dom.messageInput.value = '';
        this.autoResizeInput();
        
        // Handle file uploads first
        const filesUploaded = await this.uploadFiles();
        if (!filesUploaded) return;
        
        // Determine routing mode
        const agentMode = document.querySelector('input[name="agentMode"]:checked').value;
        
        if (agentMode === 'elite') {
            await this.handleEliteAgentRequest(message);
        } else if (agentMode === 'standard') {
            await this.handleStandardChatRequest(message);
        } else {
            await this.handleAutoRouteRequest(message);
        }
    }
    
    async handleEliteAgentRequest(message) {
        const specificAgent = this.dom.specificAgent.value;
        
        if (specificAgent) {
            await this.callSpecificAgent(message, specificAgent);
        } else {
            await this.callEliteAgentTeam(message);
        }
    }
    
    async handleStandardChatRequest(message) {
        const typingMessage = this.addMessage('...', 'ai', 'Thinking');
        
        try {
            const payload = {
                message: message,
                user_id: this.sessionId,
                task_type: "auto",
                user_tier: "free"
            };
            
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            
            this.removeMessage(typingMessage);
            const data = await response.json();
            
            if (response.ok && data.success) {
                this.addMessage(data.response, 'ai', data.model, {
                    reasoning: data.reasoning,
                    responseTime: data.response_time,
                    cost: data.cost
                });
                this.updateSystemMetrics(data);
            } else {
                this.addMessage(`Error: ${data.response || 'Request failed'}`, 'ai', 'Error');
            }
        } catch (error) {
            this.removeMessage(typingMessage);
            this.addMessage(`Network error: ${error.message}`, 'ai', 'Error');
        }
    }
    
    async handleAutoRouteRequest(message) {
        const typingMessage = this.addMessage('...', 'ai', 'Elite Router');
        
        try {
            const payload = {
                message: message,
                user_id: this.sessionId,
                task_type: "auto",
                user_tier: "free",
                business_context: this.userContext
            };
            
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            
            this.removeMessage(typingMessage);
            const data = await response.json();
            
            if (response.ok && data.success) {
                // Show routing decision if elite agents were used
                if (data.agents_used && data.agents_used.length > 0) {
                    this.showWorkflowProgress(data.agents_used, data.workflow_type);
                }
                
                this.addMessage(data.response, 'ai', data.model, {
                    reasoning: data.reasoning,
                    responseTime: data.response_time,
                    cost: data.cost,
                    agentsUsed: data.agents_used,
                    workflowType: data.workflow_type,
                    confidenceScore: data.confidence_score
                });
                
                this.updateSystemMetrics(data);
                this.logActivity('Auto-routed request', data.model);
            } else {
                this.addMessage(`Error: ${data.response || 'Request failed'}`, 'ai', 'Error');
            }
        } catch (error) {
            this.removeMessage(typingMessage);
            this.addMessage(`Network error: ${error.message}`, 'ai', 'Error');
        }
    }
    
    async callEliteAgentTeam(message) {
        const workflowMessage = this.addMessage('🔄 Analyzing request and coordinating elite agents...', 'ai', 'Elite Coordinator');
        
        try {
            const eliteRequest = {
                message: message,
                user_id: this.sessionId,
                business_type: this.userContext.business_type,
                industry: this.userContext.industry,
                target_audience: this.userContext.target_audience,
                current_revenue: this.userContext.current_revenue,
                main_challenges: this.userContext.main_challenges,
                goals: this.userContext.goals
            };
            
            const response = await fetch('/elite/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(eliteRequest)
            });
            
            this.removeMessage(workflowMessage);
            const data = await response.json();
            
            if (response.ok && data.success) {
                // Show workflow progress
                this.showWorkflowProgress(data.agents_used, data.routing_decision.workflow_type);
                
                // Format and display elite agent results
                this.displayEliteAgentResults(data);
                
                this.updateSystemMetrics({
                    response_time: data.execution_time,
                    cost: data.workflow_result.total_cost,
                    agents_used: data.agents_used
                });
                
                this.logActivity('Elite team coordination', `${data.agents_used.length} agents`);
            } else {
                this.addMessage(`Elite agent error: ${data.error || 'Request failed'}`, 'ai', 'Error');
            }
        } catch (error) {
            this.removeMessage(workflowMessage);
            this.addMessage(`Elite agent network error: ${error.message}`, 'ai', 'Error');
        }
    }
    
    async callSpecificAgent(message, agentName) {
        const agentMessage = this.addMessage(`🎯 Deploying ${agentName}...`, 'ai', 'Agent Dispatcher');
        
        try {
            const eliteRequest = {
                message: message,
                user_id: this.sessionId,
                business_type: this.userContext.business_type,
                industry: this.userContext.industry,
                target_audience: this.userContext.target_audience,
                current_revenue: this.userContext.current_revenue,
                main_challenges: this.userContext.main_challenges,
                goals: this.userContext.goals
            };
            
            const response = await fetch(`/elite/agents/${encodeURIComponent(agentName)}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(eliteRequest)
            });
            
            this.removeMessage(agentMessage);
            const data = await response.json();
            
            if (response.ok && data.success) {
                const outputText = typeof data.output === 'object' ? 
                    JSON.stringify(data.output, null, 2) : data.output;
                
                this.addMessage(outputText, 'ai', agentName, {
                    responseTime: data.execution_time,
                    cost: data.cost,
                    confidenceScore: data.confidence_score
                });
                
                this.updateSystemMetrics(data);
                this.logActivity(`${agentName} execution`, `${data.confidence_score?.toFixed(2) || 'N/A'} confidence`);
            } else {
                this.addMessage(`${agentName} error: ${data.error || 'Request failed'}`, 'ai', 'Error');
            }
        } catch (error) {
            this.removeMessage(agentMessage);
            this.addMessage(`${agentName} network error: ${error.message}`, 'ai', 'Error');
        }
    }
    
    // ============================================================================
    // UI HELPERS & DISPLAY FUNCTIONS
    // ============================================================================
    
    addMessage(content, type, agentName = '', metadata = {}) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;
        messageDiv.setAttribute('data-timestamp', Date.now());
        
        // Create agent avatar
        const avatarDiv = document.createElement('div');
        avatarDiv.className = 'agent-avatar';
        avatarDiv.textContent = type === 'user' ? '👤' : '🤖';
        
        // Create message content container
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        // Agent name
        const nameDiv = document.createElement('div');
        nameDiv.className = 'agent-name';
        nameDiv.textContent = type === 'user' ? 'You' : (agentName || 'AI Assistant');
        
        // Message text
        const textDiv = document.createElement('div');
        textDiv.className = 'message-text';
        textDiv.textContent = content;
        
        // Message metadata
        const metaDiv = document.createElement('div');
        metaDiv.className = 'message-meta';
        let metaText = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        
        if (metadata.responseTime) {
            metaText += ` • ${metadata.responseTime.toFixed(2)}s`;
        }
        if (metadata.cost) {
            metaText += ` • $${metadata.cost.toFixed(4)}`;
        }
        if (metadata.confidenceScore) {
            metaText += ` • ${Math.round(metadata.confidenceScore * 100)}% confidence`;
        }
        if (metadata.agentsUsed && metadata.agentsUsed.length > 0) {
            metaText += ` • ${metadata.agentsUsed.length} agents`;
        }
        if (metadata.reasoning) {
            metaText += ` • ${metadata.reasoning}`;
        }
        
        metaDiv.textContent = metaText;
        
        // Assemble message
        contentDiv.appendChild(nameDiv);
        contentDiv.appendChild(textDiv);
        contentDiv.appendChild(metaDiv);
        
        messageDiv.appendChild(avatarDiv);
        messageDiv.appendChild(contentDiv);
        
        this.dom.chatMessages.appendChild(messageDiv);
        this.dom.chatMessages.scrollTop = this.dom.chatMessages.scrollHeight;
        
        return messageDiv;
    }
    
    removeMessage(messageElement) {
        if (messageElement && messageElement.parentNode) {
            messageElement.parentNode.removeChild(messageElement);
        }
    }
    
    displayEliteAgentResults(data) {
        // Display routing decision first
        const routingInfo = `**🧠 Elite Routing Decision:**
- Intent: ${data.routing_decision.intent_type}
- Complexity: ${data.routing_decision.complexity_score}/10  
- Workflow: ${data.routing_decision.workflow_type}
- Agents: ${data.agents_used.join(', ')}
- Reasoning: ${data.routing_decision.reasoning}

---`;
        
        this.addMessage(routingInfo, 'ai', 'Elite Router', {
            responseTime: data.execution_time
        });
        
        // Display each agent's results
        Object.entries(data.workflow_result.results).forEach(([agentName, result]) => {
            if (result.success) {
                const agentResponse = typeof result.output === 'object' ? 
                    (result.output.response || JSON.stringify(result.output, null, 2)) : 
                    result.output;
                
                this.addMessage(agentResponse, 'ai', agentName, {
                    responseTime: result.execution_time,
                    cost: result.cost,
                    confidenceScore: result.confidence_score
                });
            } else {
                this.addMessage(`Error: ${result.output.error || 'Agent execution failed'}`, 'ai', `${agentName} (Error)`);
            }
        });
        
        // Display workflow summary
        const summary = `**🎯 Workflow Summary:**
- Total Execution Time: ${data.execution_time.toFixed(2)}s
- Total Cost: $${data.workflow_result.total_cost?.toFixed(4) || '0.0000'}
- Success Rate: ${Object.values(data.workflow_result.results).filter(r => r.success).length}/${Object.keys(data.workflow_result.results).length}
- Workflow Type: ${data.routing_decision.workflow_type}`;
        
        this.addMessage(summary, 'ai', 'Workflow Summary');
    }
    
    showWorkflowProgress(agents, workflowType) {
        if (!this.dom.workflowPanel || !this.dom.workflowSteps) return;
        
        this.dom.workflowPanel.style.display = 'block';
        this.dom.workflowSteps.innerHTML = '';
        
        agents.forEach((agentName, index) => {
            const stepDiv = document.createElement('div');
            stepDiv.className = 'workflow-step-item active';
            stepDiv.textContent = `${index + 1}. ${agentName}`;
            this.dom.workflowSteps.appendChild(stepDiv);
        });
        
        // Simulate workflow progress
        setTimeout(() => {
            const steps = this.dom.workflowSteps.querySelectorAll('.workflow-step-item');
            steps.forEach((step, index) => {
                setTimeout(() => {
                    step.classList.remove('active');
                    step.classList.add('completed');
                }, index * 1000);
            });
        }, 1000);
    }
    
    handleModeChange() {
        const selectedMode = document.querySelector('input[name="agentMode"]:checked').value;
        
        if (selectedMode === 'elite') {
            this.dom.eliteSelector.style.display = 'block';
            this.dom.modelStatus.textContent = 'Mode: Elite Direct';
        } else {
            this.dom.eliteSelector.style.display = 'none';
            this.dom.modelStatus.textContent = selectedMode === 'auto' ? 'Mode: Auto Route' : 'Mode: Standard';
        }
    }
    
    // ============================================================================
    // FILE HANDLING
    // ============================================================================
    
    handleFiles(files) {
        for (const file of files) {
            if (!this.attachedFiles.some(f => f.name === file.name)) {
                this.attachedFiles.push(file);
            }
        }
        this.renderAttachments();
    }
    
    renderAttachments() {
        this.dom.fileAttachments.innerHTML = '';
        this.attachedFiles.forEach((file, index) => {
            const attachmentEl = document.createElement('div');
            attachmentEl.className = 'file-attachment';
            attachmentEl.innerHTML = `
                📄 ${file.name}
                <button onclick="eliteAgent.removeFile(${index})">&times;</button>
            `;
            this.dom.fileAttachments.appendChild(attachmentEl);
        });
    }
    
    removeFile(index) {
        this.attachedFiles.splice(index, 1);
        this.renderAttachments();
    }
    
    async uploadFiles() {
        if (this.attachedFiles.length === 0) return true;
        
        this.addMessage(`📁 Uploading ${this.attachedFiles.length} file(s)...`, 'ai', 'File Manager');
        
        for (const file of this.attachedFiles) {
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                const response = await fetch('/upload-file', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) throw new Error(`Upload failed for ${file.name}`);
                
                const result = await response.json();
                this.addMessage(`✅ ${result.filename} uploaded successfully`, 'ai', 'File Manager');
            } catch (error) {
                this.addMessage(`❌ Error uploading ${file.name}: ${error.message}`, 'ai', 'File Manager');
                return false;
            }
        }
        
        this.attachedFiles = [];
        this.renderAttachments();
        return true;
    }
    
    // ============================================================================
    // UTILITY FUNCTIONS
    // ============================================================================
    
    autoResizeInput() {
        this.dom.messageInput.style.height = 'auto';
        this.dom.messageInput.style.height = Math.min(this.dom.messageInput.scrollHeight, 120) + 'px';
    }
    
    updateSessionStatus() {
        this.dom.sessionStatus.textContent = `Session: ${this.sessionId.split('-')[2] || 'Unknown'}`;
        this.dom.agentsStatus.textContent = `Agents: ${Object.keys(this.agentStatuses).length}`;
    }
    
    updateSystemMetrics(data) {
        this.systemMetrics.totalRequests++;
        
        if (data.response_time) {
            this.dom.timeStatus.textContent = `Time: ${data.response_time.toFixed(1)}s`;
        }
        
        if (data.cost) {
            const currentCost = parseFloat(this.dom.costStatus.textContent.replace('Cost: $', '')) || 0;
            this.dom.costStatus.textContent = `Cost: $${(currentCost + data.cost).toFixed(4)}`;
        }
        
        if (data.agents_used && data.agents_used.length > 0) {
            this.dom.agentsStatus.textContent = `Agents: ${data.agents_used.length} active`;
        }
    }
    
    logActivity(activity, detail = '') {
        const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        const activityItem = document.createElement('div');
        activityItem.className = 'activity-item';
        activityItem.innerHTML = `
            <div class="activity-time">${timestamp}</div>
            <div class="activity-desc">${activity}${detail ? ` - ${detail}` : ''}</div>
        `;
        
        this.dom.activityLog.insertBefore(activityItem, this.dom.activityLog.firstChild);
        
        // Keep only last 10 activities
        while (this.dom.activityLog.children.length > 10) {
            this.dom.activityLog.removeChild(this.dom.activityLog.lastChild);
        }
    }
    
    showNotification(message, type = 'info') {
        // Simple notification system
        const notification = document.createElement('div');
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: ${type === 'success' ? '#22c55e' : type === 'error' ? '#ef4444' : '#4f46e5'};
            color: white;
            padding: 12px 20px;
            border-radius: 8px;
            z-index: 1000;
            animation: slideIn 0.3s ease-out;
        `;
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.remove();
        }, 3000);
    }
}

// ============================================================================
// GLOBAL FUNCTIONS FOR QUICK ACTIONS
// ============================================================================

function insertQuickMessage(message) {
    eliteAgent.dom.messageInput.value = message;
    eliteAgent.autoResizeInput();
    eliteAgent.dom.messageInput.focus();
}

// ============================================================================
// INITIALIZATION
// ============================================================================

// Initialize the Elite Agent Interface when DOM is ready
let eliteAgent;

document.addEventListener('DOMContentLoaded', () => {
    eliteAgent = new EliteAgentInterface();
    
    // Add some welcome animations
    setTimeout(() => {
        eliteAgent.logActivity('System initialized', 'All agents ready');
    }, 1000);
    
    console.log('🔥 Elite Agent Command Center initialized');
});

// Export for global access
window.eliteAgent = eliteAgent;