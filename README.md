# Phishing Email Detection

## Overview
This project aims to detect phishing emails using machine learning models. The system analyzes email content and metadata to classify emails as either phishing or legitimate. The model is trained using a dataset of phishing and non-phishing emails and employs feature extraction techniques to enhance classification accuracy.

## Features
- **Machine Learning Models:** Utilizes Support Vector Machine (SVM) and Random Forest for classification.
- **Feature Engineering:** Extracts key features such as email headers, body text, and suspicious links.
- **Evaluation Metrics:** F1-score, precision-recall analysis for model selection.
- **Scalability:** Can be extended with deep learning for enhanced detection.

## Installation
To set up and run the project locally:

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/phishing-email-detection.git
   cd phishing-email-detection
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the training script:
   ```sh
   python train_model.py
   ```
4. Test the model:
   ```sh
   python test_model.py
   ```

## Dataset
The project uses a dataset containing phishing and non-phishing emails. It includes:
- Email headers
- Email body text
- Presence of suspicious URLs

## Usage
After training, the model can be used to classify emails:
```sh
python classify_email.py --email "path_to_email.txt"
```

## Future Enhancements
- Integration with deep learning models for improved accuracy.
- Real-time email scanning and classification.
- API deployment using FastAPI.

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests.

## License
This project is licensed under the MIT License.

