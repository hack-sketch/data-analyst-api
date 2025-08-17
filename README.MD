# 🤖 Data Analyst Agent API

[![Deploy](https://github.com/yourusername/data-analyst-api/actions/workflows/deploy.yml/badge.svg)](https://github.com/yourusername/data-analyst-api/actions/workflows/deploy.yml)
[![Docker](https://img.shields.io/badge/docker-ready-blue)](https://github.com/yourusername/data-analyst-api/pkgs/container/data-analyst-api)

A powerful AI-powered data analysis API that can source, prepare, analyze, and visualize any data using LLMs and machine learning techniques.

## 🚀 Quick Start

### One-Click Deployment

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template/your-template)
[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

### Local Development

```bash
# Clone the repository
git clone https://github.com/yourusername/data-analyst-api.git
cd data-analyst-api

# Install dependencies
pip install -r requirements.txt

# Run the API
python app.py
```

API will be available at `http://localhost:8000`

## 📡 API Usage

### Endpoint
```
POST https://your-api-url.com/api/
```

### Example Request
```bash
curl -X POST https://your-api-url.com/api/ \
  -F "questions.txt=@questions.txt" \
  -F "data.csv=@your_data.csv"
```

### Sample questions.txt
```
What are the basic statistics of this dataset?
Show me the correlation between variables
Create a histogram of the numeric columns
What trends can you identify in the data?
Perform a regression analysis
```

## 🔧 Features

- **Multi-format Data Loading**: CSV, Excel, JSON, TXT files
- **External Data Sources**: Yahoo Finance, public APIs
- **Advanced Analytics**: Correlation, regression, statistical tests
- **Visualizations**: Charts returned as base64 images
- **Natural Language Processing**: Answer questions in plain English
- **Scalable Architecture**: Docker-ready, cloud-native design

## 📊 Supported Analysis Types

| Analysis Type | Description | Example Question |
|---------------|-------------|------------------|
| Basic Stats | Data profiling, summary statistics | "What are the basic statistics?" |
| Correlation | Relationship analysis between variables | "Show correlation between columns" |
| Regression | Predictive modeling | "Predict sales based on marketing spend" |
| Time Series | Trend analysis over time | "What's the trend in stock prices?" |
| Distribution | Data distribution analysis | "Show histogram of ages" |
| Financial | Stock market analysis | "Analyze AAPL stock performance" |

## 🏗️ Project Structure

```
data-analyst-api/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── Dockerfile            # Container configuration
├── .github/
│   └── workflows/
│       └── deploy.yml    # CI/CD pipeline
├── tests/
│   └── test_api.py       # API tests
├── examples/
│   ├── questions.txt     # Sample questions
│   └── sample_data.csv   # Sample dataset
└── README.md            # This file
```

## 🚢 Deployment Options

### 1. Railway (Recommended)
1. Fork this repository
2. Connect to [Railway](https://railway.app)
3. Deploy automatically from GitHub

### 2. Render
1. Fork this repository  
2. Connect to [Render](https://render.com)
3. Create new Web Service from GitHub

### 3. Heroku
```bash
# Install Heroku CLI
heroku create your-app-name
git push heroku main
```

### 4. Google Cloud Run
```bash
gcloud run deploy data-analyst-api \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### 5. Docker
```bash
# Build and run locally
docker build -t data-analyst-api .
docker run -p 8000:8000 data-analyst-api

# Or pull from GitHub Container Registry
docker run -p 8000:8000 ghcr.io/yourusername/data-analyst-api:latest
```

## 🧪 Testing

### Health Check
```bash
curl https://your-api-url.com/health
```

### Run Tests
```bash
python -m pytest tests/ -v
```

### Load Testing
```bash
# Install Apache Bench
apt-get install apache2-utils

# Test concurrent requests
ab -n 100 -c 10 https://your-api-url.com/health
```

## 📝 API Response Format

```json
{
  "results": [
    {
      "question": "What is the correlation between variables?",
      "answer": {
        "correlation_matrix": {...}
      },
      "visualization": "base64_encoded_image",
      "data_source": "data.csv",
      "status": "success"
    }
  ],
  "total_questions": 1,
  "processed_questions": 1,
  "data_files_loaded": ["data.csv"],
  "processing_time_seconds": 2.5,
  "timestamp": "2025-08-17T12:00:00Z",
  "status": "success"
}
```

## 🔒 Security

- Input validation and sanitization
- File size limits (50MB max)
- Secure filename handling
- Rate limiting ready
- Container security best practices

## ⚡ Performance

- 5-minute timeout per request
- Memory-efficient data processing
- Automatic timeout management
- Concurrent request handling
- Base64 image optimization

## 🛠️ Development

### Adding New Analysis Types
1. Extend `DataAnalystAgent` class in `app.py`
2. Add new methods for your analysis
3. Update question parsing logic
4. Test with sample data

### Adding New Data Sources
1. Implement new fetch methods
2. Add authentication if needed
3. Update data loading pipeline
4. Document in README

## 📊 Example Use Cases

### Financial Analysis
```bash
echo "What is the trend of AAPL stock prices over the last year?" > questions.txt
curl -X POST https://your-api-url.com/api/ -F "questions.txt=@questions.txt"
```

### Sales Data Analysis
```bash
echo -e "What are the sales trends?\nWhich products are most profitable?\nPredict next quarter sales" > questions.txt
curl -X POST https://your-api-url.com/api/ \
  -F "questions.txt=@questions.txt" \
  -F "sales_data.csv=@sales.csv"
```

### Marketing Analytics
```bash
echo -e "Show correlation between marketing spend and sales\nWhich channels are most effective?" > questions.txt
curl -X POST https://your-api-url.com/api/ \
  -F "questions.txt=@questions.txt" \
  -F "marketing_data.xlsx=@marketing.xlsx"
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- 📧 **Issues**: [GitHub Issues](https://github.com/yourusername/data-analyst-api/issues)
- 📖 **Documentation**: [Wiki](https://github.com/yourusername/data-analyst-api/wiki)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/yourusername/data-analyst-api/discussions)

## 🏆 Acknowledgments

- Built with Flask, pandas, scikit-learn, and matplotlib
- Inspired by modern data science workflows
- Designed for production scalability

---

⭐ **Star this repository if you find it useful!** ⭐
