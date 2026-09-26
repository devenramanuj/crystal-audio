import os
import gradio as gr
import soundfile as sf
from df.enhance import enhance, init_df, load_audio, save_audio

# મોડેલ પ્રારંભિક લોડ કરો
print("DeepFilterNet મોડેલ લોડ થઈ રહ્યું છે...")
model, df_state, _ = init_df()
print("મોડેલ તૈયાર છે!")

def remove_noise(audio_path):
    if audio_path is None:
        return None
    
    # આઉટપુટ ફાઈલનો પાથ
    output_path = "cleaned_audio.wav"
    
    try:
        # ૧. મોડેલના સેમ્પલ રેટ સાથે ઓડિયો લોડ કરો (DeepFilterNet 48kHz વાપરે છે)
        audio, _ = load_audio(audio_path, sr=df_state.sr())
        
        # ૨. AI એન્હાન્સમેન્ટ પ્રોસેસ
        enhanced_audio = enhance(model, df_state, audio)
        
        # ૩. ક્લીન થયેલો ઓડિયો સેવ કરો
        save_audio(output_path, enhanced_audio, sr=df_state.sr())
        
        return output_path
    except Exception as e:
        print(f"Error during processing: {e}")
        return None

# Gradio UI સેટઅપ
custom_css = """
footer {visibility: hidden}
.gradio-container {max-width: 700px !important; margin: auto;}
"""

demo = gr.Interface(
    fn=remove_noise,
    inputs=gr.Audio(sources=["upload", "microphone"], type="filepath", label="ઓરિજિનલ ઓડિયો (અપલોડ કરો અથવા રેકોર્ડ કરો)"),
    outputs=gr.Audio(type="filepath", label="સાફ થયેલો ઓડિયો (Clean AI Audio)"),
    title="🎙️ AI Audio Noise Remover & Enhancer",
    description="ડીપ લર્નિંગ (DeepFilterNet) મોડેલ આધારિત ટૂલ. પંખા, ટ્રાફિક અને અનિયમિત બેકગ્રાઉન્ડ અવાજો દૂર કરી સ્ટુડિયો જેવો ક્લિયર અવાજ મેળવો.",
    css=custom_css,
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch()
