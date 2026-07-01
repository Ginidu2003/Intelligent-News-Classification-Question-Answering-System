# 📰 Intelligent News Classification & Question Answering System

A fine-tuned **DistilBERT** model for classifying Sri Lankan news articles into 5 categories with an interactive **Gradio web app** featuring a Question Answering capability.

## 🎯 Project Overview

This project combines two powerful NLP tasks:

1. **News Classification**: Automatically classify news articles into 5 categories
   - Business
   - Opinion
   - Political Gossip
   - Sports
   - World News

2. **Question Answering**: Ask questions about news articles and get intelligent answers powered by RoBERTa

## 🌟 Features

- ✅ Fine-tuned DistilBERT model for Sri Lankan news classification
- ✅ Support for batch processing via CSV uploads
- ✅ Interactive Question Answering on news content
- ✅ Beautiful category distribution visualization
- ✅ CSV output with classification results
- ✅ Confidence scores for Q&A responses
- ✅ Text preprocessing with lemmatization and tokenization

## 📊 Web App Usage

### Accessing the Web App

The interactive web app is deployed on **Hugging Face Spaces** and accessible through the following URL:

🔗 **[Launch Web App](https://huggingface.co/spaces/Ginidu2003/News_Classifier)**

### Tab 1: 📊 News Classification

#### How to Use:

1. **Prepare Your CSV File**
   - Your CSV file must contain a column named **`content`** with the news articles
   - Example structure:
     ```
     id,content,date
     1,"Article text here..."
     2,"Another article..."
     ```

2. **Upload the CSV File**
   - Click on the **📤 Upload your CSV file** box
   - Select a CSV file from your computer
   - You can use sample files from the `Data Files` folder:
     - `evaluation.csv`
     - `output.csv` (already classified examples)

3. **Run Classification**
   - Click the **🚀 Classify News** button
   - The model will process all articles and classify them

4. **View Results**
   - **Status**: Shows the number of successfully classified articles
   - **📊 Category Distribution Chart**: Visual representation of how many articles fall into each category
   - **📥 Download output.csv**: Contains your original data + a new `class` column with predictions

#### Example CSV from Data Files

You can download sample data from the `Data Files` folder:
- `output.csv` - Pre-classified example showing expected output format

### Tab 2: ❓ Question Answering

#### How to Use:

1. **Paste News Content**
   - In the **📝 Paste News Content** field, paste or type the full news article
   - You can use any news text or classified articles from your results

2. **Enter Your Question**
   - In the **❓ Your Question** field, type your question
   - Example questions:
     - "What is the main topic?"
     - "Who are the key people mentioned?"
     - "When did the event occur?"

3. **Get Answer**
   - Click the **🔍 Get Answer** button
   - The model will extract the answer from the news content and provide a confidence score

4. **View Results**
   - **Answer**: The extracted answer from the article
   - **Confidence**: The model's confidence percentage (higher is better)

#### Example Workflow:

```
News Content: "The Central Bank announced a new interest rate policy..."
Question: "What did the Central Bank announce?"
Answer: "a new interest rate policy"
Confidence: 94.23%
```

## 📁 Project Structure

```
Intelligent-News-Classification-Question-Answering-System/
├── app1.py                              # Gradio web app (main application)
├── Text_Analysis_Project.ipynb          # Jupyter notebook with model training & analysis
├── README.md                            # This file
└── Data Files/
    ├── evaluation.csv                   # Test set for classification
    ├── output.csv                       # Example output with classifications
    ├── Business.xlsx                    # Business news articles
    ├── Opinion.xlsx                     # Opinion news articles
    ├── Political_gossip.xlsx            # Political gossip news articles
    ├── Sports.xlsx                      # Sports news articles
    ├── World_news.xlsx                  # World news articles
    ├── Daily_Mirror_News.xlsx           # Full dataset from Daily Mirror
    └── _Preprocessed_Daily_Mirror_News.xlsx  # Preprocessed version
```

## 🤖 Models Used

### Classification Model
- **Base Model**: DistilBERT (distilbert-base-uncased)
- **Fine-tuned Version**: `Ginidu2003/Distilbert-Base-News-classifier`
- **Categories**: 5 (Business, Opinion, Political Gossip, Sports, World News)
- **Framework**: Hugging Face Transformers

### Question Answering Model
- **Base Model**: RoBERTa-base fine-tuned on SQuAD 2.0
- **Model**: `deepset/roberta-base-squad2`
- **Capability**: Extractive QA (finds answers within provided text)

## 🔧 Technical Stack

- **Deep Learning**: PyTorch, Transformers (Hugging Face)
- **Web Framework**: Gradio
- **Data Processing**: Pandas, NumPy
- **NLP**: NLTK, Tokenization, Lemmatization
- **Visualization**: Matplotlib
- **Language**: Python 3.x



### Classification Output CSV

The output CSV includes:
- All original columns from your input CSV
- **`class`**: New column containing the predicted category
  - Values: Business, Opinion, Political_Gossip, Sports, World_News

Example output structure:
```csv
id,content,date,class
1,"Article text...",Sports
2,"Another text...",Business
```

### Category Distribution Chart

A beautiful bar chart showing:
- Count of articles in each category
- 5 distinct color-coded bars
- Values displayed on top of each bar for easy reference

## ⚠️ Important Notes

1. **CSV Column Requirement**: Your CSV file MUST have a column named `content` for classification to work

2. **Text Quality**: The model performs best with:
   - Complete news articles
   - Proper English text
   - Articles related to news content

3. **QA Limitations**:
   - The Q&A model extracts answers from provided text (extractive QA)
   - It cannot generate answers outside the provided content
   - Works best with clear, well-written articles

4. **Data Privacy**: All processing happens in the Gradio app. Files are processed in real-time and not stored permanently.

## 📊 Model Performance

- **Classification Accuracy**: Trained on diverse Sri Lankan news articles
- **QA Confidence**: Varies by article quality and question clarity (typically 70-95%)
- **Processing Speed**: Fast inference due to DistilBERT (lightweight model)

## 💡 Tips for Best Results

### For Classification:
1. Use CSV files with clearly written news articles
2. Ensure the `content` column contains complete article text
3. Batch processing multiple articles improves efficiency

### For Question Answering:
1. Paste the complete article for context
2. Ask specific questions that can be answered from the article
3. Longer, more detailed articles give better answers

## 📚 Additional Resources

- **Model Card**: [Distilbert-Base-News-classifier](https://huggingface.co/Ginidu2003/Distilbert-Base-News-classifier)
- **Dataset**: Daily Mirror Sri Lankan news articles
- **Gradio Documentation**: [https://gradio.app](https://gradio.app)
- **Transformers Library**: [https://huggingface.co/docs/transformers](https://huggingface.co/docs/transformers)





## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report issues
- Suggest improvements
- Submit pull requests

## 📧 Contact & Support

For questions or support, please open an issue on the GitHub repository.

---

**Happy Classifying! 🚀**
