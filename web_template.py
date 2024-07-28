css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''
bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://static.vecteezy.com/system/resources/previews/005/192/268/original/chatbot-notification-color-icon-chat-bot-face-with-exclamation-mark-important-message-artificial-conversational-entity-virtual-assistant-announcement-isolated-illustration-vector.jpg" style="max-height: 80px; max-width: 80px; border="20" >
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://static.vecteezy.com/system/resources/previews/015/890/239/non_2x/humanoid-icon-color-outline-vector.jpg" style="max-height: 100px; max-width: 80px; border="20" >
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''