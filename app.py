import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load the model securely from the same directory
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'adaboostmodel.pkl')
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# Attractive, self-contained HTML layout using Tailwind CSS
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Developer Analytics Predictor</title>
    <script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"></script>
    <style>
        body {
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
            min-height: 100vh;
        }
    </style>
</head>
<body class="text-slate-100 font-sans antialiased flex flex-col justify-center items-center py-12 px-4">

    <div class="max-w-4xl w-full bg-slate-900/60 backdrop-blur-md border border-slate-800 rounded-2xl shadow-2xl p-8 md:p-12">
        
        <div class="text-center mb-10">
            <h1 class="text-3xl md:text-4xl font-extrabold tracking-tight bg-gradient-to-r from-indigo-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
                Developer Metrics Analytics
            </h1>
            <p class="text-slate-400 mt-2 text-sm md:text-base">
                Input your metrics below to evaluate prediction outcomes from the AdaBoost Classifier.
            </p>
        </div>

        {% if prediction_text %}
        <div class="mb-8 p-4 bg-emerald-500/10 border border-emerald-500/30 rounded-xl text-center shadow-lg shadow-emerald-500/5 animate-fade-in">
            <span class="text-emerald-400 font-bold text-lg block">{{ prediction_text }}</span>
        </div>
        {% endif %}

        {% if error_text %}
        <div class="mb-8 p-4 bg-rose-500/10 border border-rose-500/30 rounded-xl text-center shadow-lg shadow-rose-500/5">
            <span class="text-rose-400 font-semibold text-sm block">{{ error_text }}</span>
        </div>
        {% endif %}

        <form action="/predict" method="POST" class="space-y-6">
            <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                
                <div>
                    <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Hours Coding</label>
                    <input type="number" step="any" name="Hours_Coding" required placeholder="e.g. 6.5"
                        class="w-full bg-slate-950/80 border border-slate-800 rounded-lg px-4 py-3 text-slate-100 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition">
                </div>

                <div>
                    <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">AI Usage Hours</label>
                    <input type="number" step="any" name="AI_Usage_Hours" required placeholder="e.g. 2"
                        class="w-full bg-slate-950/80 border border-slate-800 rounded-lg px-4 py-3 text-slate-100 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition">
                </div>

                <div>
                    <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Lines of Code</label>
                    <input type="number" step="any" name="Lines_of_Code" required placeholder="e.g. 150"
                        class="w-full bg-slate-950/80 border border-slate-800 rounded-lg px-4 py-3 text-slate-100 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition">
                </div>

                <div>
                    <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Commits</label>
                    <input type="number" step="any" name="Commits" required placeholder="e.g. 5"
                        class="w-full bg-slate-950/80 border border-slate-800 rounded-lg px-4 py-3 text-slate-100 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition">
                </div>

                <div>
                    <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Bugs Reported</label>
                    <input type="number" step="any" name="Bugs_Reported" required placeholder="e.g. 1"
                        class="w-full bg-slate-950/80 border border-slate-800 rounded-lg px-4 py-3 text-slate-100 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition">
                </div>

                <div>
                    <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Sleep Hours</label>
                    <input type="number" step="any" name="Sleep_Hours" required placeholder="e.g. 7.5"
                        class="w-full bg-slate-950/80 border border-slate-800 rounded-lg px-4 py-3 text-slate-100 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition">
                </div>

                <div>
                    <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Distractions</label>
                    <input type="number" step="any" name="Distractions" required placeholder="e.g. 3"
                        class="w-full bg-slate-950/80 border border-slate-800 rounded-lg px-4 py-3 text-slate-100 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition">
                </div>

                <div>
                    <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Cognitive Load</label>
                    <input type="number" step="any" name="Cognitive_Load" required placeholder="Scale value"
                        class="w-full bg-slate-950/80 border border-slate-800 rounded-lg px-4 py-3 text-slate-100 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition">
                </div>

                <div>
                    <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Stress Level</label>
                    <input type="number" step="any" name="Stress_Level" required placeholder="Scale value"
                        class="w-full bg-slate-950/80 border border-slate-800 rounded-lg px-4 py-3 text-slate-100 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition">
                </div>

            </div>

            <div class="pt-4">
                <button type="submit" 
                    class="w-full bg-gradient-to-r from-indigo-500 to-purple-600 hover:from-indigo-600 hover:to-purple-700 text-white font-bold py-3.5 px-6 rounded-xl shadow-xl hover:shadow-indigo-500/20 transition duration-200 transform active:scale-[0.99] cursor-pointer">
                    Generate Analytics Prediction
                </button>
            </div>
        </form>
    </div>

</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from form submission
        features = [
            float(request.form.get('Hours_Coding', 0)),
            float(request.form.get('AI_Usage_Hours', 0)),
            float(request.form.get('Lines_of_Code', 0)),
            float(request.form.get('Commits', 0)),
            float(request.form.get('Bugs_Reported', 0)),
            float(request.form.get('Sleep_Hours', 0)),
            float(request.form.get('Distractions', 0)),
            float(request.form.get('Cognitive_Load', 0)),
            float(request.form.get('Stress_Level', 0))
        ]
        
        # Structure data array for Scikit-Learn
        final_features = [np.array(features)]
        prediction = model.predict(final_features)[0]
        
        return render_template_string(HTML_TEMPLATE, prediction_text=f'Predicted Result: {prediction}')
    
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, error_text=f"Deployment Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
