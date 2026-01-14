# Intelligent-Human-Assistant

In part 1, we aimed to make the command line interface more usable by allowing users to enter natural language queries. In this part, we will add a new capability—making computer use easier with Voice Control (on iOS) or Voice Access (on Android). Let's call it Voice User Interface (VUI) to be agnostic to iOS or Android. Like command lines, VUIs are very rigid—users must memorize commands, and any minor errors in command syntax are discarded by the VUI system.

Our goal is to clean up users' utterances in real-time using a fine-tuned small language model (e.g., Gemma 3 270M). It takes the raw user command and returns an output that the VUI expects at that time.

To make this system practical, we use two devices: a phone where the VUI is running, and a laptop where the LLM chatbot is running. For now, the chatbot takes user commands via text input, cleans them up, and reads them out loud so that the phone (nearby) listens to the command and acts upon it.

Refer to your OS-specific VUI documentation:
iOS: https://support.apple.com/guide/iphone/use-voice-control-iph2c21a3c88/iosLinks to an external site.
Android: https://support.google.com/accessibility/android/answer/6151848?hl=enLinks to an external site.
 
Architecture
Similar to Part 1, the system includes:
An MCP server hosting several tools
An MCP client hosting a locally-run LLM and offering a terminal-based text input
However, unlike Part 1, the MCP client now:
Runs through a graphical user interface based on Streamlit (see HW-RAG)
Has speech output capability (through the pyttsx3 library)
Additionally, the MCP server has a new tool that internally calls the smaller, fine-tuned LLM (let's call it Voice Command Converter or VCC). You need to fine-tune this model.

Users can type two types of commands: /echo [command] and /vc [command]. /echo commands are immediately read out by the interface.

 /vc commands will go through the entire MCP pipeline (UI -> MCP client -> MCP server -> VCC)
