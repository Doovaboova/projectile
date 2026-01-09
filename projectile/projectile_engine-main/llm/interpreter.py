import os
import google.generativeai as genai

# ✅ Secure API key loading
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found. Set it as an environment variable.")

genai.configure(api_key=GEMINI_API_KEY)

MODEL = "gemini-1.5-flash"

def interpret_results(true_params, inferred_params, noise):
    prompt = f"""
You are a physicist analyzing a projectile motion inference experiment.

TRUE PARAMETERS:
- Initial velocity: {true_params['v0']} m/s
- Launch angle: {true_params['angle']} degrees
- Drag coefficient: {true_params['k']}

INFERRED PARAMETERS:
- Initial velocity: {inferred_params['v0']:.2f} m/s
- Launch angle: {inferred_params['angle']:.2f} degrees
- Drag coefficient: {inferred_params['k']:.4f}

Sensor noise level: {noise}

Explain:
1. Accuracy of the inference
2. Likely sources of error
3. Why drag is hardest to estimate
4. Whether the result is physically reasonable

Write clearly in scientific but intuitive language.
Limit to ~150 words.
"""

    model = genai.GenerativeModel(MODEL)
    response = model.generate_content(prompt)

    return response.text
