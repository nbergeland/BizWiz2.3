# 🍗 BizWiz 2.3: Real-Time Location Intelligence Platform

**Advanced restaurant site selection and market analysis with dynamic data integration**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Dash](https://img.shields.io/badge/dashboard-Dash-green.svg)](https://dash.plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![LOGO](https://github.com/user-attachments/assets/40ac4db5-7211-4d52-b197-ef9c95cfffdd)


## 🚀 Overview

BizWiz 2.3 is a comprehensive location intelligence platform designed for restaurant chain expansion. It combines real-time data collection, machine learning, and interactive visualization to identify optimal restaurant locations across 219+ US cities.

### Key Features

- **🏙️ Multi-City Analysis**: Pre-configured support for 219 major US cities
- **⚡ Real-Time Data Loading**: On-demand data collection with live progress tracking
- **🤖 ML-Powered Predictions**: Revenue forecasting using advanced Random Forest models
- **🗺️ Interactive Mapping**: Live location visualization with competitor analysis
- **📊 Dynamic Dashboard**: Real-time analytics with multiple specialized views
- **🎯 Competitor Intelligence**: Automated competitor mapping and distance analysis
- **📈 Market Insights**: Comprehensive demographic and commercial analysis

## 📁 Project Structure

```
BizWiz2.3/
├── 🏗️ Core Configuration
│   ├── city_config.py              # Multi-city configuration system
│   ├── generate_usa_cities.py      # City database generator
│   └── usa_city_configs.yaml       # Generated city configurations
│
├── 🔄 Dynamic Data System
│   ├── dynamic_data_loader.py      # Real-time data collection engine
│   └── dynamic_dashboard.py        # Interactive dashboard application
│
├── 🛠️ Setup & Utilities
│   ├── setup_dynamic.py           # Automated setup script
│   ├── debug_city_configs.py      # Configuration debugging
│   ├── requirements_dynamic.txt   # Python dependencies
│   └── launch_dynamic_dashboard.py # Dashboard launcher (auto-generated)
│
└── 📚 Documentation
    └── README.md                   # This file
```

## 🚀 Quick Start

### 1. **One-Command Setup**
```bash
python setup_dynamic.py
```

### 2. **Launch Dashboard**
```bash
python launch_dynamic_dashboard.py
```

### 3. **Open Browser**
Navigate to: **http://127.0.0.1:8051**

## 🔧 Manual Installation

### Prerequisites
- Python 3.8+
- 2GB+ RAM
- Internet connection for data APIs

### Step-by-Step Setup

1. **Clone/Download the repository**
   ```bash
   cd BizWiz2.3
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements_dynamic.txt
   ```

3. **Generate city database**
   ```bash
   python generate_usa_cities.py
   ```

4. **Launch dashboard**
   ```bash
   python dynamic_dashboard.py
   ```

## 🎯 Core Features

### 🏙️ Multi-City Configuration System
- **219 Pre-configured Cities**: All major US cities with population >50k
- **Adaptive Analysis Grids**: Population-based grid spacing for optimal coverage
- **Market Intelligence**: State-specific competitor and demographic data
- **Extensible Design**: Easy addition of new cities and markets

### ⚡ Real-Time Data Loading
- **On-Demand Processing**: Data loads only when city is selected
- **Progress Tracking**: Live progress bars with ETA estimates
- **Smart Caching**: 1-hour cache to prevent redundant API calls
- **Fallback Systems**: Synthetic data when APIs unavailable

### 🤖 Machine Learning Engine
- **Revenue Prediction**: Random Forest models with demographic features
- **Feature Engineering**: Income-age interactions, competition pressure
- **Model Validation**: Cross-validation with performance metrics
- **Real-Time Training**: Models trained fresh for each city analysis

### 📊 Interactive Dashboard

#### **🗺️ Live Location Map**
- Interactive city-wide location analysis
- Competitor overlay with distance calculations
- Revenue potential heatmaps
- Performance statistics cards

#### **📈 Real-Time Analytics**
- Revenue distribution analysis
- Income vs. revenue correlations
- Competition impact visualization
- Traffic pattern analysis

#### **🏆 Opportunity Ranking**
- Top 20 revenue opportunities
- Sortable data tables with metrics
- Performance percentile analysis
- Location recommendation scoring

#### **🔬 Model Intelligence**
- Model performance metrics (R², MAE)
- Feature importance analysis
- Real-time data freshness indicators
- Cross-validation results

#### **💡 Market Insights**
- Automated market overview generation
- Competitive landscape analysis
- Demographic trend identification
- Strategic recommendations

## 🎛️ Configuration

### City Configuration
Cities are automatically configured with:
- **Geographic Bounds**: Adaptive sizing based on population
- **Demographics**: Income, age, and population density factors
- **Market Data**: Universities, major employers, state info
- **Competition**: Primary competitors and search terms

### API Integration
For real data (optional):
```python
# In dynamic_data_loader.py
CENSUS_API_KEY = "your_census_key"  # For demographic data
GOOGLE_PLACES_KEY = "your_google_key"  # For competitor mapping
```

## 📊 Data Sources

### **Real-Time APIs** (with fallbacks)
- **US Census Bureau**: Demographic and economic data
- **Google Places**: Competitor location mapping
- **Traffic APIs**: Accessibility and traffic patterns
- **Commercial Data**: Zoning and real estate information

### **Synthetic Data Generation**
- Population-based demographic modeling
- Distance-weighted traffic scoring
- Random competitor distribution
- Commercial viability scoring

## 🔬 Technical Architecture

### **Frontend**
- **Dash + Bootstrap**: Responsive web interface
- **Plotly**: Interactive visualizations and maps
- **Real-time Updates**: Progress tracking and live data

### **Backend**
- **Async Processing**: Non-blocking data collection
- **Threading**: Background data loading
- **Caching**: Intelligent data persistence
- **Error Handling**: Graceful API failure management

### **Data Pipeline**
1. **Grid Generation**: Adaptive geographic sampling
2. **Data Collection**: Parallel API calls with rate limiting
3. **Feature Engineering**: Demographic and spatial features
4. **Model Training**: Real-time ML pipeline
5. **Visualization**: Interactive dashboard rendering

## 🚀 Advanced Usage

### Adding New Cities
```python
from city_config import CityConfigManager

manager = CityConfigManager()
# Cities automatically detected from generate_usa_cities.py
```

### Custom Analysis
```python
from dynamic_data_loader import load_city_data_on_demand

# Load data for specific city
city_data = await load_city_data_on_demand(
    city_id="austin_tx",
    force_refresh=True
)
```

### Debugging
```bash
python debug_city_configs.py  # Debug city configurations
python test_city_loading.py   # Test data loading system
```

## 🔧 Troubleshooting

### Common Issues

**Port Already in Use**
```bash
lsof -ti:8051 | xargs kill -9  # Kill process on port 8051
```

**Missing Dependencies**
```bash
pip install --upgrade -r requirements_dynamic.txt
```

**City Not Found**
```bash
python debug_city_configs.py  # Check available cities
```

**Dashboard Won't Load**
```bash
python test_dashboard_direct.py  # Test dashboard startup
```

### Performance Optimization
- **Grid Spacing**: Adjust `grid_spacing` in city configs for faster processing
- **Cache Duration**: Modify `cache_timeout` in data loader
- **Batch Size**: Tune API batch sizes for optimal performance

## 🎯 Use Cases

### **Restaurant Chain Expansion**
- Identify high-revenue potential locations
- Analyze competitive landscape
- Optimize market entry strategies
- Risk assessment and location scoring

### **Market Research**
- Demographic trend analysis
- Competition mapping and analysis
- Commercial real estate evaluation
- Traffic and accessibility studies

### **Investment Analysis**
- Location-based ROI predictions
- Market saturation assessment
- Growth opportunity identification
- Portfolio optimization

## 📈 Roadmap

### **Current Version (2.3)**
- ✅ 219 US cities supported
- ✅ Real-time data integration
- ✅ Interactive dashboard
- ✅ ML-powered predictions

### **Future Enhancements**
- 🔄 International city support
- 🔄 Advanced ML models (XGBoost, Neural Networks)
- 🔄 Real estate API integration
- 🔄 Mobile application
- 🔄 Multi-tenant architecture
- 🔄 API endpoint exposure

## 🤝 Contributing

### Development Setup
```bash
git clone <repository>
cd BizWiz2.3
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements_dynamic.txt
```

### Code Style
- **PEP 8**: Python code formatting
- **Type Hints**: Function annotations encouraged
- **Docstrings**: Comprehensive documentation
- **Testing**: pytest for unit tests

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙋‍♂️ Support

### Documentation
- **Setup Issues**: Run `python setup_dynamic.py`
- **Configuration**: Check `debug_city_configs.py`
- **Performance**: Review technical architecture section

### Contact
- **Issues**: GitHub Issues
- **Questions**: GitHub Discussions
- **Email**: Contact@BBSLLC.AI
## 🎖️ Credits

### Technologies Used
- **[Dash](https://dash.plotly.com/)**: Web application framework
- **[Plotly](https://plotly.com/)**: Interactive visualizations
- **[scikit-learn](https://scikit-learn.org/)**: Machine learning
- **[pandas](https://pandas.pydata.org/)**: Data manipulation
- **[aiohttp](https://aiohttp.readthedocs.io/)**: Async HTTP client

### Data Sources
- **US Census Bureau**: Demographic data
- **Google Places API**: Business location data
- **OpenStreetMap**: Geographic data

---

**Made with ❤️ for smarter business decisions**

*BizWiz 2.3 - Where data meets location intelligence*
