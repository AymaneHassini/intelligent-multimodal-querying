"""
Streamlit application for Unified NL2SQL & GIQ Chatbot.
"""
import streamlit as st
from chains.basic_chain import invoke_basic_chain
from chains.advanced_chain import invoke_advanced_chain
from utils.arabic import is_query_arabic, translate_to_arabic

def main():
    """Main application entry point."""
    st.title("Unified NL2SQL & GIQ Chatbot")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_user_prompt" not in st.session_state:
        st.session_state.last_user_prompt = ""
    if "advanced_clicked_for" not in st.session_state:
        st.session_state.advanced_clicked_for = ""
    if "giq_running" not in st.session_state:
        st.session_state.giq_running = False
    
    # Mode selection
    mode = st.radio("Select Query Mode", ["Basic (NL-to-SQL)", "Advanced (GIQ)"])
    
    # Show conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # New user prompt
    if user_q := st.chat_input("Enter your query:"):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_q})
        st.session_state.last_user_prompt = user_q
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_q)
        
        # Generate response
        with st.spinner("Generating response..."):
            if mode == "Basic (NL-to-SQL)":
                # Use basic NL-to-SQL chain
                raw = invoke_basic_chain(user_q, st.session_state.messages)
            else:
                # Use advanced integrated pipeline
                raw = invoke_advanced_chain(user_q, st.session_state.messages)
            
            # Translate if the prompt was in Arabic
            if is_query_arabic(user_q):
                response = translate_to_arabic(raw)
            else:
                response = raw
            
            # Add assistant message to history
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Display assistant message
            with st.chat_message("assistant"):
                st.markdown(response)
    
    # Show "Try Advanced Reasoning" button only after Basic
    if mode == "Basic (NL-to-SQL)":
        st.session_state.show_advanced_button = True
    
    # Advanced-after-Basic button
    if (mode == "Basic (NL-to-SQL)" and st.session_state.get("show_advanced_button", False)):
        already = (st.session_state.last_user_prompt == st.session_state.advanced_clicked_for)
        btn = st.button(
            "🔍 Try Advanced Reasoning",
            disabled=already or st.session_state.giq_running
        )
        
        if btn and not already:
            st.session_state.advanced_clicked_for = st.session_state.last_user_prompt
            st.session_state.giq_running = True
            
            with st.spinner("Running advanced reasoning..."):
                raw_adv = invoke_advanced_chain(
                    st.session_state.last_user_prompt,
                    st.session_state.messages
                )
                
                if is_query_arabic(st.session_state.last_user_prompt):
                    adv_response = translate_to_arabic(raw_adv)
                else:
                    adv_response = raw_adv
                
                st.session_state.messages.append(
                    {"role": "assistant", "content": adv_response}
                )
                
                with st.chat_message("assistant"):
                    st.markdown(adv_response)
                    
            st.session_state.giq_running = False

if __name__ == "__main__":
    main()