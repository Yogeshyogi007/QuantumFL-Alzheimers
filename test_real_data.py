import torch
from pathlib import Path
import sys

# Add the project root to Python path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from models.cnn_model import AlzheimerCNN
from inference.predict import preprocess_mri

def test_all_models():
    """Test all models with real data samples."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test samples - Control vs Alzheimer's
    test_samples = [
        ("Control Sample 1", "data/raw/disc1/OAS1_0001_MR1/RAW/OAS1_0001_MR1_mpr-1_anon_sag_66.gif"),
        ("Control Sample 2", "data/raw/disc1/OAS1_0002_MR1/RAW/OAS1_0002_MR1_mpr-1_anon_sag_66.gif"),
        ("Control Sample 3", "data/raw/disc1/OAS1_0003_MR1/RAW/OAS1_0003_MR1_mpr-1_anon_sag_66.gif"),
        ("Alzheimer's Sample 1", "data/raw/disc1/OAS1_0028_MR1/RAW/OAS1_0028_MR1_mpr-1_anon_sag_66.gif"),
    ]
    
    # Models to test
    models = [
        ("Best Model (99.81%)", "models/best_alzheimers_cnn.pth"),
        ("Balanced Model", "models/balanced_alzheimers_cnn.pth"),
        ("Conservative Model", "models/conservative_alzheimers_cnn.pth"),
    ]
    
    print("🧠 COMPREHENSIVE MODEL TESTING WITH REAL DATA")
    print("="*80)
    
    results = {}
    
    for model_name, model_path in models:
        print(f"\n🔬 Testing {model_name}")
        print("-" * 50)
        
        try:
            # Load model
            model = AlzheimerCNN()
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            
            model_results = []
            
            for sample_name, sample_path in test_samples:
                try:
                    # Preprocess and predict
                    x = preprocess_mri(sample_path).to(device)
                    
                    with torch.no_grad():
                        output = model(x)
                        prob = torch.softmax(output, dim=1)[0, 1].item()
                    
                    risk_level = "HIGH RISK" if prob > 0.5 else "LOW RISK"
                    confidence = max(prob, 1-prob)
                    
                    print(f"  {sample_name}: {prob:.4f} ({risk_level}, {confidence:.1%} confidence)")
                    model_results.append((sample_name, prob, risk_level, confidence))
                    
                except Exception as e:
                    print(f"  ❌ Error processing {sample_name}: {e}")
                    model_results.append((sample_name, None, "ERROR", 0))
            
            results[model_name] = model_results
            
        except Exception as e:
            print(f"❌ Error loading {model_name}: {e}")
            results[model_name] = []
    
    # Summary comparison
    print(f"\n📊 SUMMARY COMPARISON")
    print("="*80)
    print(f"{'Sample':<20} {'Best Model':<12} {'Balanced':<12} {'Conservative':<12}")
    print("-" * 80)
    
    for i, (sample_name, _) in enumerate(test_samples):
        best_prob = results["Best Model (99.81%)"][i][1] if results["Best Model (99.81%)"][i][1] is not None else "N/A"
        balanced_prob = results["Balanced Model"][i][1] if results["Balanced Model"][i][1] is not None else "N/A"
        conservative_prob = results["Conservative Model"][i][1] if results["Conservative Model"][i][1] is not None else "N/A"
        
        print(f"{sample_name:<20} {best_prob:<12} {balanced_prob:<12} {conservative_prob:<12}")
    
    # Analysis
    print(f"\n🎯 ANALYSIS")
    print("="*80)
    
    # Check if models can differentiate between control and Alzheimer's samples
    for model_name, model_results in results.items():
        if len(model_results) >= 4:
            control_probs = [r[1] for r in model_results[:3] if r[1] is not None]  # First 3 are control
            alzheimers_probs = [r[1] for r in model_results[3:] if r[1] is not None]  # Last 1 is Alzheimer's
            
            if control_probs and alzheimers_probs:
                avg_control = sum(control_probs) / len(control_probs)
                avg_alzheimers = sum(alzheimers_probs) / len(alzheimers_probs)
                
                print(f"{model_name}:")
                print(f"  Average Control Probability: {avg_control:.4f}")
                print(f"  Average Alzheimer's Probability: {avg_alzheimers:.4f}")
                print(f"  Difference: {avg_alzheimers - avg_control:.4f}")
                
                if avg_alzheimers > avg_control:
                    print(f"  ✅ Model correctly differentiates (Alzheimer's > Control)")
                else:
                    print(f"  ❌ Model does not differentiate correctly")
                print()

if __name__ == "__main__":
    test_all_models() 