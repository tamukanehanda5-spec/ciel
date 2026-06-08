"""
Ciel - AI Voice Assistant
Main application entry point
"""

import os
import sys
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Import configuration
from config import Config

# Initialize OpenAI client
client = OpenAI(api_key=Config.OPENAI_API_KEY)


class CielAssistant:
    """Main Ciel AI Assistant class"""
    
    def __init__(self):
        """Initialize the assistant"""
        self.config = Config
        self.config.validate()
        self.conversation_history = []
        self.name = Config.ASSISTANT_NAME
        
    def listen(self):
        """
        Listen to user's voice input and convert to text
        Using OpenAI's Whisper API
        """
        try:
            import sounddevice as sd
            import numpy as np
            from scipy.io import wavfile
        except ImportError:
            print("❌ Audio libraries not installed. Installing...")
            os.system("pip install sounddevice scipy")
            import sounddevice as sd
            import numpy as np
            from scipy.io import wavfile
        
        print("\n🎤 Listening... (speak now, will record for 5 seconds)")
        
        try:
            # Record audio
            duration = 5  # seconds
            sample_rate = Config.MIC_SAMPLE_RATE
            
            audio = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype=np.int16
            )
            sd.wait()
            
            # Save temporarily
            temp_file = "temp_audio.wav"
            wavfile.write(temp_file, sample_rate, audio)
            
            # Transcribe using Whisper
            print("🔄 Transcribing...")
            with open(temp_file, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="en"
                )
            
            user_input = transcript.text
            
            # Clean up temp file
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
            print(f"📝 You said: {user_input}")
            return user_input
            
        except Exception as e:
            print(f"❌ Error listening: {e}")
            print("📝 Enter text instead:")
            return input("> ")
    
    def think(self, user_message):
        """
        Process user message with AI and generate response
        """
        print(f"\n🧠 {self.name} is thinking...")
        
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Keep only recent messages to stay within token limit
        if len(self.conversation_history) > Config.CONVERSATION_HISTORY_LENGTH:
            self.conversation_history = self.conversation_history[-Config.CONVERSATION_HISTORY_LENGTH:]
        
        try:
            # Get response from GPT
            response = client.chat.completions.create(
                model=Config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": Config.SYSTEM_PROMPT},
                    *self.conversation_history
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            assistant_message = response.choices[0].message.content
            
            # Add to history
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })
            
            return assistant_message
            
        except Exception as e:
            print(f"❌ Error thinking: {e}")
            return "I encountered an error processing your request. Please try again."
    
    def speak(self, text):
        """
        Convert text response to speech and play it
        Using OpenAI's TTS API
        """
        print(f"\n🔊 {self.name}: {text}\n")
        
        try:
            # Generate speech
            print("🔄 Generating speech...")
            speech_file = "temp_speech.mp3"
            
            response = client.audio.speech.create(
                model="tts-1",
                voice=Config.VOICE_NAME,
                text=text
            )
            
            response.stream_to_file(speech_file)
            
            # Play audio
            print("▶️ Playing audio...")
            self.play_audio(speech_file)
            
            # Clean up
            if os.path.exists(speech_file):
                os.remove(speech_file)
                
        except Exception as e:
            print(f"❌ Error speaking: {e}")
            print("Falling back to text display")
    
    def play_audio(self, filepath):
        """Play audio file"""
        try:
            import platform
            import subprocess
            
            system = platform.system()
            
            if system == "Darwin":  # macOS
                subprocess.run(["afplay", filepath])
            elif system == "Linux":
                subprocess.run(["aplay", filepath])
            elif system == "Windows":
                import winsound
                winsound.PlaySound(filepath, winsound.SND_FILENAME)
        except Exception as e:
            print(f"⚠️ Could not play audio: {e}")
    
    def run(self):
        """Main conversation loop"""
        print(f"\n{'='*50}")
        print(f"👋 Welcome to {self.name}")
        print(f"{'='*50}")
        print("\nHow to use:")
        print("  - Speak to ask questions or request help")
        print("  - Type 'exit' or 'quit' to end conversation")
        print("  - Type 'text' to switch to text input mode")
        print(f"\n{self.name} is ready to help! 🚀\n")
        
        use_voice = True
        
        while True:
            try:
                # Get user input
                if use_voice:
                    user_input = self.listen()
                else:
                    user_input = input("\n📝 You: ")
                
                # Check for exit commands
                if user_input.lower() in ["exit", "quit", "bye"]:
                    print(f"\n👋 {self.name}: Goodbye! Have a great day!")
                    break
                
                if user_input.lower() == "text":
                    use_voice = not use_voice
                    mode = "text" if use_voice is False else "voice"
                    print(f"✓ Switched to {mode} mode")
                    continue
                
                if not user_input.strip():
                    continue
                
                # Get AI response
                response = self.think(user_input)
                
                # Speak response
                if use_voice:
                    self.speak(response)
                else:
                    print(f"\n🔊 {self.name}: {response}\n")
                    
            except KeyboardInterrupt:
                print(f"\n\n👋 {self.name}: Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                print("Let's try again...")


def main():
    """Main entry point"""
    try:
        assistant = CielAssistant()
        assistant.run()
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        print("\n📖 Setup Instructions:")
        print("1. Create a .env file (copy from .env.example)")
        print("2. Add your OpenAI API key from https://platform.openai.com/api-keys")
        print("3. Run this script again")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Fatal Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
