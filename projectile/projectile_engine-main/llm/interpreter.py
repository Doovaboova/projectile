import os
import google.generativeai as genai

MODEL = "gemini-1.5-flash"

def interpret_results(true_params, inferred_params, noise):
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return (
            "⚠️ **LLM interpretation unavailable.**\n\n"
            "The `GEMINI_API_KEY` environment variable is not set.\n"
            "Numerical inference results are still valid."
        )

    genai.configure(api_key=api_key)

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
3. Why drag is difficult to estimate
4. Whether the result is physically reasonable

Limit response to ~150 words.
"""

    model = genai.GenerativeModel(MODEL)
    response = model.generate_content(prompt)

    return response.text
