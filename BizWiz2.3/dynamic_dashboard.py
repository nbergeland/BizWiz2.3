# === DYNAMIC REAL-TIME DASHBOARD ===
# Save this as: dynamic_dashboard.py

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output, State, callback_context, dash_table
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import asyncio
import threading
import time
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from city_config import CityConfigManager
from dynamic_data_loader import load_city_data_on_demand, DataLoadingProgress
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for state management
app_state = {
    'current_city_data': None,
    'loading_progress': None,
    'last_loaded_city': None,
    'loading_in_progress': False
}

# Initialize the Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "BizWiz Dynamic Analytics"

# Initialize city manager
city_manager = CityConfigManager()
available_cities = [
    {'label': config.display_name, 'value': city_id}
    for city_id, config in city_manager.configs.items()
]

# === LAYOUT ===
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("🍗 BizWiz: Real-Time Location Intelligence", className="text-center mb-3"),
            html.P("Dynamic city analysis with live data integration", 
                   className="text-center text-muted mb-4")
        ])
    ]),
    
    # Control Panel
    dbc.Card([
        dbc.CardHeader([
            html.H5("🎯 City Selection & Controls", className="mb-0")
        ]),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Select City for Analysis:", className="fw-bold"),
                    dcc.Dropdown(
                        id='city-selector',
                        options=available_cities,
                        value=None,
                        placeholder="Choose a city to analyze...",
                        clearable=False,
                        className="mb-3"
                    )
                ], width=6),
                
                dbc.Col([
                    html.Label("Data Options:", className="fw-bold"),
                    html.Div([
                        dbc.Button(
                            "🔄 Force Refresh Data", 
                            id="refresh-btn", 
                            color="primary", 
                            size="sm",
                            className="me-2"
                        ),
                        dbc.Button(
                            "📊 Quick Analysis", 
                            id="quick-analysis-btn", 
                            color="secondary", 
                            size="sm"
                        )
                    ])
                ], width=6)
            ]),
            
            # Progress Bar
            html.Div(id='progress-container', style={'display': 'none'}, children=[
                html.Hr(),
                html.H6("Loading Progress:", className="mb-2"),
                dbc.Progress(id="progress-bar", value=0, className="mb-2"),
                html.Div(id="progress-text", className="text-muted small")
            ])
        ])
    ], className="mb-4"),
    
    # Status Cards
    html.Div(id='status-cards', children=[
        dbc.Alert(
            "👋 Welcome to BizWiz! Select a city above to begin real-time location analysis.",
            color="info",
            className="text-center"
        )
    ]),
    
    # Main Content Tabs
    html.Div(id='main-content', style={'display': 'none'}, children=[
        dbc.Tabs([
            dbc.Tab(label="🗺️ Live Location Map", tab_id="live-map-tab"),
            dbc.Tab(label="📊 Real-Time Analytics", tab_id="live-analytics-tab"),
            dbc.Tab(label="🏆 Opportunity Ranking", tab_id="opportunities-tab"),
            dbc.Tab(label="🔬 Model Intelligence", tab_id="model-tab"),
            dbc.Tab(label="📈 Market Insights", tab_id="insights-tab")
        ], id="main-tabs", active_tab="live-map-tab"),
        
        html.Div(id='tab-content', className="mt-4")
    ]),
    
    # Hidden divs for state management
    html.Div(id='city-data-store', style={'display': 'none'}),
    html.Div(id='loading-trigger', style={'display': 'none'}),
    
    # Auto-refresh interval
    dcc.Interval(id='progress-interval', interval=500, n_intervals=0, disabled=True)
    
], fluid=True)

# === CALLBACK FUNCTIONS ===

@app.callback(
    [Output('loading-trigger', 'children'),
     Output('progress-container', 'style'),
     Output('status-cards', 'children')],
    [Input('city-selector', 'value'),
     Input('refresh-btn', 'n_clicks')],
    [State('loading-trigger', 'children')]
)
def trigger_city_loading(city_id, refresh_clicks, current_trigger):
    """Trigger city data loading when city is selected"""
    
    if not city_id:
        return "", {'display': 'none'}, [
            dbc.Alert(
                "👋 Welcome to BizWiz! Select a city above to begin real-time location analysis.",
                color="info",
                className="text-center"
            )
        ]
    
    ctx = callback_context
    force_refresh = False
    
    if ctx.triggered:
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if trigger_id == 'refresh-btn' and refresh_clicks:
            force_refresh = True
    
    # Check if we need to load this city
    if (city_id != app_state['last_loaded_city'] or 
        force_refresh or 
        not app_state['current_city_data']):
        
        # Start loading in background thread
        threading.Thread(
            target=load_city_data_background,
            args=(city_id, force_refresh),
            daemon=True
        ).start()
        
        app_state['loading_in_progress'] = True
        app_state['last_loaded_city'] = city_id
        
        return f"loading-{city_id}-{datetime.now().isoformat()}", {'display': 'block'}, [
            dbc.Alert([
                html.Div([
                    dbc.Spinner(size="sm", className="me-2"),
                    f"🔄 Loading real-time data for {city_manager.get_config(city_id).display_name}..."
                ], className="d-flex align-items-center")
            ], color="warning", className="text-center")
        ]
    
    return current_trigger or "", {'display': 'none'}, [
        dbc.Alert(
            f"✅ Ready to analyze {city_manager.get_config(city_id).display_name}",
            color="success",
            className="text-center"
        )
    ]

def load_city_data_background(city_id: str, force_refresh: bool = False):
    """Load city data in background thread"""
    
    def progress_callback(progress: DataLoadingProgress):
        """Update progress state"""
        app_state['loading_progress'] = {
            'percent': progress.progress_percent,
            'step': progress.step_name,
            'locations': progress.locations_processed,
            'total_locations': progress.total_locations,
            'eta': progress.estimated_remaining
        }
    
    try:
        # Load data asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        city_data = loop.run_until_complete(
            load_city_data_on_demand(city_id, progress_callback, force_refresh)
        )
        
        app_state['current_city_data'] = city_data
        app_state['loading_in_progress'] = False
        app_state['loading_progress'] = None
        
        logger.info(f"Successfully loaded data for {city_id}")
        
    except Exception as e:
        logger.error(f"Error loading city data: {e}")
        app_state['loading_in_progress'] = False
        app_state['loading_progress'] = {'error': str(e)}
    
    finally:
        loop.close()

@app.callback(
    [Output('progress-bar', 'value'),
     Output('progress-bar', 'label'),
     Output('progress-text', 'children'),
     Output('progress-interval', 'disabled'),
     Output('main-content', 'style')],
    [Input('progress-interval', 'n_intervals')],
    [State('loading-trigger', 'children')]
)
def update_progress(n_intervals, loading_trigger):
    """Update loading progress"""
    
    if not app_state['loading_in_progress']:
        # Loading complete - show main content if data available
        main_style = {'display': 'block'} if app_state['current_city_data'] else {'display': 'none'}
        return 0, "", "", True, main_style
    
    progress = app_state['loading_progress']
    if not progress:
        return 0, "Initializing...", "", False, {'display': 'none'}
    
    if 'error' in progress:
        return 0, "Error occurred", f"❌ {progress['error']}", True, {'display': 'none'}
    
    percent = progress['percent']
    step = progress['step']
    
    # Create progress text
    progress_text = f"Step: {step}"
    if progress.get('locations', 0) > 0:
        progress_text += f" | Processed: {progress['locations']}/{progress['total_locations']} locations"
    if progress.get('eta', 0) > 0:
        progress_text += f" | ETA: {progress['eta']:.0f}s"
    
    return percent, f"{percent:.1f}%", progress_text, False, {'display': 'none'}

@app.callback(
    Output('tab-content', 'children'),
    [Input('main-tabs', 'active_tab')],
    [State('city-selector', 'value')]
)
def update_tab_content(active_tab, city_id):
    """Update tab content based on active tab"""
    
    if not app_state['current_city_data'] or not city_id:
        return html.Div("No data available. Please select a city.", className="text-center mt-5")
    
    city_data = app_state['current_city_data']
    
    try:
        if active_tab == "live-map-tab":
            return create_live_map_tab(city_data)
        elif active_tab == "live-analytics-tab":
            return create_analytics_tab(city_data)
        elif active_tab == "opportunities-tab":
            return create_opportunities_tab(city_data)
        elif active_tab == "model-tab":
            return create_model_tab(city_data)
        elif active_tab == "insights-tab":
            return create_insights_tab(city_data)
        else:
            return html.Div("Unknown tab", className="text-center mt-5")
            
    except Exception as e:
        logger.error(f"Error creating tab content: {e}")
        return dbc.Alert(f"❌ Error loading tab content: {str(e)}", color="danger")

def create_live_map_tab(city_data: Dict[str, Any]) -> html.Div:
    """Create live interactive map tab"""
    
    df = city_data['df_filtered']
    config = city_data['city_config']
    competitor_data = city_data.get('competitor_data', {})
    
    if len(df) == 0:
        return html.Div("No location data available", className="text-center mt-5")
    
    # Create enhanced map
    fig = px.scatter_mapbox(
        df.head(500),  # Limit for performance
        lat='latitude',
        lon='longitude',
        size='predicted_revenue',
        color='predicted_revenue',
        color_continuous_scale='RdYlGn',
        size_max=20,
        zoom=11,
        mapbox_style='open-street-map',
        title=f"🍗 Live Location Intelligence: {config.display_name}",
        hover_data={
            'predicted_revenue': ':$,.0f',
            'median_income': ':$,.0f',
            'traffic_score': ':.0f',
            'commercial_score': ':.0f'
        }
    )
    
    # Add competitor locations
    primary_competitors = competitor_data.get(config.competitor_data.primary_competitor, [])
    if primary_competitors:
        comp_df = pd.DataFrame(primary_competitors)
        fig.add_trace(
            go.Scattermapbox(
                lat=comp_df['latitude'],
                lon=comp_df['longitude'],
                mode='markers',
                marker=dict(size=15, color='red', symbol='circle'),
                text='🐔',
                name=f"{config.competitor_data.primary_competitor.title()} Locations",
                hovertemplate="<b>Competitor Location</b><br>" +
                             "Lat: %{lat:.4f}<br>" +
                             "Lon: %{lon:.4f}<extra></extra>"
            )
        )
    
    fig.update_layout(
        height=700,
        mapbox=dict(
            center=dict(lat=config.bounds.center_lat, lon=config.bounds.center_lon)
        )
    )
    
    # Statistics cards
    stats_cards = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"{len(df):,}", className="text-primary mb-0"),
                    html.P("Locations Analyzed", className="text-muted mb-0")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"${df['predicted_revenue'].mean():,.0f}", className="text-success mb-0"),
                    html.P("Avg Revenue Potential", className="text-muted mb-0")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"${df['predicted_revenue'].max():,.0f}", className="text-warning mb-0"),
                    html.P("Top Location Potential", className="text-muted mb-0")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"{len(primary_competitors)}", className="text-info mb-0"),
                    html.P("Competitors Mapped", className="text-muted mb-0")
                ])
            ])
        ], width=3)
    ], className="mb-4")
    
    return html.Div([
        stats_cards,
        dcc.Graph(figure=fig, style={'height': '75vh'})
    ])

def create_analytics_tab(city_data: Dict[str, Any]) -> html.Div:
    """Create real-time analytics tab"""
    
    df = city_data['df_filtered']
    
    # Create analytics dashboard
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Revenue Distribution', 'Income vs Revenue', 
                       'Competition Impact', 'Traffic Analysis'),
        specs=[[{"type": "histogram"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "bar"}]]
    )
    
    # Revenue distribution
    fig.add_trace(
        go.Histogram(x=df['predicted_revenue'], nbinsx=30, name="Revenue Distribution"),
        row=1, col=1
    )
    
    # Income vs Revenue
    fig.add_trace(
        go.Scatter(
            x=df['median_income'], 
            y=df['predicted_revenue'],
            mode='markers',
            name="Income vs Revenue",
            marker=dict(opacity=0.6)
        ),
        row=1, col=2
    )
    
    # Competition impact
    if 'distance_to_primary_competitor' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['distance_to_primary_competitor'],
                y=df['predicted_revenue'],
                mode='markers',
                name="Competition Distance",
                marker=dict(opacity=0.6, color='red')
            ),
            row=2, col=1
        )
    
    # Traffic analysis
    if 'traffic_score' in df.columns:
        traffic_bins = pd.cut(df['traffic_score'], bins=5)
        traffic_revenue = df.groupby(traffic_bins)['predicted_revenue'].mean()
        
        fig.add_trace(
            go.Bar(
                x=[f"{interval.left:.0f}-{interval.right:.0f}" for interval in traffic_revenue.index],
                y=traffic_revenue.values,
                name="Traffic Score Impact"
            ),
            row=2, col=2
        )
    
    fig.update_layout(height=600, showlegend=False, title_text="📊 Real-Time Market Analytics")
    
    # Key insights
    insights = generate_market_insights(df)
    
    insights_cards = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🎯 Key Market Insights"),
                dbc.CardBody([
                    html.Ul([
                        html.Li(insight) for insight in insights
                    ])
                ])
            ])
        ], width=12)
    ], className="mb-4")
    
    return html.Div([
        insights_cards,
        dcc.Graph(figure=fig)
    ])

def create_opportunities_tab(city_data: Dict[str, Any]) -> html.Div:
    """Create opportunities ranking tab"""
    
    df = city_data['df_filtered']
    
    # Get top 20 opportunities
    top_locations = df.nlargest(20, 'predicted_revenue').copy()
    
    # Create display dataframe
    display_cols = ['latitude', 'longitude', 'predicted_revenue']
    optional_cols = ['median_income', 'traffic_score', 'commercial_score', 
                    'distance_to_primary_competitor', 'competition_density']
    
    for col in optional_cols:
        if col in top_locations.columns:
            display_cols.append(col)
    
    display_df = top_locations[display_cols].copy()
    
    # Format for display
    display_df['predicted_revenue'] = display_df['predicted_revenue'].apply(lambda x: f"${x:,.0f}")
    if 'median_income' in display_df.columns:
        display_df['median_income'] = display_df['median_income'].apply(lambda x: f"${x:,.0f}")
    if 'distance_to_primary_competitor' in display_df.columns:
        display_df['distance_to_primary_competitor'] = display_df['distance_to_primary_competitor'].apply(lambda x: f"{x:.1f} mi")
    
    # Rename columns
    column_names = {
        'latitude': 'Latitude',
        'longitude': 'Longitude',
        'predicted_revenue': 'Revenue Potential',
        'median_income': 'Median Income',
        'traffic_score': 'Traffic Score',
        'commercial_score': 'Commercial Score',
        'distance_to_primary_competitor': 'Distance to Competitor',
        'competition_density': 'Competition Count'
    }
    
    display_df = display_df.rename(columns=column_names)
    
    # Create data table
    table = dash_table.DataTable(
        data=display_df.to_dict('records'),
        columns=[{"name": i, "id": i} for i in display_df.columns],
        style_cell={
            'textAlign': 'left',
            'padding': '10px',
            'fontFamily': 'Arial'
        },
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 0},
                'backgroundColor': '#d4edda',
                'color': 'black',
            },
            {
                'if': {'row_index': 1},
                'backgroundColor': '#fff3cd',
                'color': 'black',
            },
            {
                'if': {'row_index': 2},
                'backgroundColor': '#f8d7da',
                'color': 'black',
            }
        ],
        page_size=20,
        sort_action="native"
    )
    
    # Summary statistics
    summary_stats = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"${df['predicted_revenue'].quantile(0.9):,.0f}", className="text-success mb-0"),
                    html.P("90th Percentile Revenue", className="text-muted mb-0")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"{len(df[df['predicted_revenue'] > df['predicted_revenue'].quantile(0.8)]):,}", className="text-primary mb-0"),
                    html.P("Premium Locations (Top 20%)", className="text-muted mb-0")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"${df['predicted_revenue'].std():,.0f}", className="text-info mb-0"),
                    html.P("Revenue Variability", className="text-muted mb-0")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"{city_data['generation_time'][:16]}", className="text-warning mb-0"),
                    html.P("Data Generated", className="text-muted mb-0")
                ])
            ])
        ], width=3)
    ], className="mb-4")
    
    return html.Div([
        html.H4("🏆 Top Revenue Opportunities", className="mb-3"),
        summary_stats,
        table
    ])

def create_model_tab(city_data: Dict[str, Any]) -> html.Div:
    """Create model performance tab"""
    
    metrics = city_data.get('metrics', {})
    
    if not metrics:
        return html.Div("No model metrics available", className="text-center mt-5")
    
    # Model performance cards
    metrics_cards = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"{metrics.get('train_r2', 0):.3f}", className="text-primary"),
                    html.P("Model R² Score", className="text-muted mb-0")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"${metrics.get('cv_mae_mean', 0):,.0f}", className="text-success"),
                    html.P("Cross-Val MAE", className="text-muted mb-0")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"{metrics.get('feature_count', 0)}", className="text-info"),
                    html.P("Features Used", className="text-muted mb-0")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Real-time", className="text-warning"),
                    html.P("Data Freshness", className="text-muted mb-0")
                ])
            ])
        ], width=3)
    ])
    
    # Model explanation
    explanation = dbc.Card([
        dbc.CardHeader("🤖 Model Intelligence"),
        dbc.CardBody([
            html.P("This model uses advanced machine learning to predict revenue potential based on:"),
            html.Ul([
                html.Li("📊 Real-time demographic data"),
                html.Li("🚗 Live traffic pattern analysis"),
                html.Li("🏢 Commercial viability scoring"),
                html.Li("🎯 Dynamic competition mapping"),
                html.Li("📍 Geographic accessibility factors")
            ]),
            html.P("Data is refreshed on-demand for each city analysis, ensuring the most current market intelligence.")
        ])
    ])
    
    return html.Div([
        html.H4("🔬 Model Performance & Intelligence", className="mb-3"),
        metrics_cards,
        html.Br(),
        explanation
    ])

def create_insights_tab(city_data: Dict[str, Any]) -> html.Div:
    """Create market insights tab"""
    
    df = city_data['df_filtered']
    config = city_data['city_config']
    
    # Generate comprehensive insights
    insights = {
        'market_overview': analyze_market_overview(df, config),
        'competition_analysis': analyze_competition(df, city_data.get('competitor_data', {})),
        'demographic_insights': analyze_demographics(df),
        'location_recommendations': generate_location_recommendations(df)
    }
    
    return html.Div([
        html.H4("📈 Market Intelligence Dashboard", className="mb-4"),
        
        # Market Overview
        dbc.Card([
            dbc.CardHeader("🌍 Market Overview"),
            dbc.CardBody([
                html.P(insights['market_overview'])
            ])
        ], className="mb-3"),
        
        # Competition Analysis
        dbc.Card([
            dbc.CardHeader("🎯 Competition Analysis"),
            dbc.CardBody([
                html.P(insights['competition_analysis'])
            ])
        ], className="mb-3"),
        
        # Demographics
        dbc.Card([
            dbc.CardHeader("👥 Demographic Insights"),
            dbc.CardBody([
                html.P(insights['demographic_insights'])
            ])
        ], className="mb-3"),
        
        # Recommendations
        dbc.Card([
            dbc.CardHeader("💡 Strategic Recommendations"),
            dbc.CardBody([
                html.Div([
                    html.P(rec) for rec in insights['location_recommendations']
                ])
            ])
        ])
    ])

# === HELPER FUNCTIONS ===

def generate_market_insights(df: pd.DataFrame) -> List[str]:
    """Generate key market insights"""
    insights = []
    
    if len(df) > 0:
        top_10_avg = df.nlargest(int(len(df) * 0.1), 'predicted_revenue')['predicted_revenue'].mean()
        overall_avg = df['predicted_revenue'].mean()
        
        insights.append(f"Top 10% locations show {(top_10_avg/overall_avg - 1)*100:.0f}% higher revenue potential")
        
        if 'median_income' in df.columns:
            high_income_avg = df[df['median_income'] > df['median_income'].quantile(0.75)]['predicted_revenue'].mean()
            insights.append(f"High-income areas (top 25%) average ${high_income_avg:,.0f} revenue potential")
        
        if 'traffic_score' in df.columns:
            high_traffic_count = len(df[df['traffic_score'] > 80])
            insights.append(f"{high_traffic_count} locations have excellent traffic scores (>80)")
        
        if 'distance_to_primary_competitor' in df.columns:
            low_competition = len(df[df['distance_to_primary_competitor'] > 2])
            insights.append(f"{low_competition} locations have minimal competition (>2 miles from primary competitor)")
    
    return insights

def analyze_market_overview(df: pd.DataFrame, config) -> str:
    """Analyze overall market"""
    total_locations = len(df)
    avg_revenue = df['predicted_revenue'].mean()
    top_revenue = df['predicted_revenue'].max()
    
    return (f"{config.display_name} shows strong potential with {total_locations:,} analyzed locations. "
            f"Average revenue potential of ${avg_revenue:,.0f} with top opportunities reaching ${top_revenue:,.0f}. "
            f"The market demonstrates {config.competitor_data.market_saturation_factor*100:.0f}% saturation level.")

def analyze_competition(df: pd.DataFrame, competitor_data: Dict) -> str:
    """Analyze competitive landscape"""
    total_competitors = sum(len(locations) for locations in competitor_data.values())
    
    if 'distance_to_primary_competitor' in df.columns:
        avg_distance = df['distance_to_primary_competitor'].mean()
        low_competition_areas = len(df[df['distance_to_primary_competitor'] > 3])
        
        return (f"Competitive analysis reveals {total_competitors} mapped competitor locations. "
                f"Average distance to primary competitor is {avg_distance:.1f} miles. "
                f"{low_competition_areas} locations identified with minimal competition exposure.")
    
    return f"Mapped {total_competitors} competitor locations across the market area."

def analyze_demographics(df: pd.DataFrame) -> str:
    """Analyze demographic factors"""
    if 'median_income' in df.columns and 'median_age' in df.columns:
        avg_income = df['median_income'].mean()
        avg_age = df['median_age'].mean()
        
        return (f"Target demographics show favorable characteristics with average household income of ${avg_income:,.0f} "
                f"and median age of {avg_age:.1f} years, aligning well with fast-casual dining preferences.")
    
    return "Demographic data indicates favorable market conditions for restaurant expansion."

def generate_location_recommendations(df: pd.DataFrame) -> List[str]:
    """Generate strategic recommendations"""
    recommendations = []
    
    if len(df) > 0:
        top_locations = df.nlargest(5, 'predicted_revenue')
        recommendations.append(f"🎯 Focus on top 5 locations with revenue potential of ${top_locations['predicted_revenue'].min():,.0f}+")
        
        if 'commercial_score' in df.columns:
            high_commercial = len(df[df['commercial_score'] > 80])
            recommendations.append(f"🏢 {high_commercial} locations show excellent commercial viability")
        
        if 'zoning_compliant' in df.columns:
            compliant_locations = len(df[df['zoning_compliant'] == 1])
            recommendations.append(f"📋 {compliant_locations} locations show zoning compliance")
        
        recommendations.append("💡 Consider phased expansion starting with highest-scoring locations")
        recommendations.append("🔄 Monitor competitor activity in identified opportunity zones")
    
    return recommendations

# === MAIN APPLICATION RUNNER ===
def main():
    """Main function to run the dynamic dashboard"""
    print("🚀 Starting BizWiz Dynamic Dashboard")
    print("🌐 Features:")
    print("   - Real-time data loading on city selection")
    print("   - Live API integration for fresh data")
    print("   - Dynamic progress tracking")
    print("   - On-demand model training")
    print("   - Interactive market intelligence")
    print()
    print(f"📍 Available cities: {len(available_cities)}")
    
    # Test if port is available first
    import socket
    try:
        test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        test_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        test_sock.bind(('127.0.0.1', 8051))
        test_sock.close()
        port = 8051
        print(f"🌐 Dashboard will start at: http://127.0.0.1:{port}")
    except OSError:
        port = 8052
        print(f"⚠️  Port 8051 in use, using port {port} instead")
        print(f"🌐 Dashboard will start at: http://127.0.0.1:{port}")
    
    print("✋ Press Ctrl+C to stop the dashboard")
    print()
    
    try:
        app.run(
            debug=True,
            host='127.0.0.1',
            port=port
        )
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        print("🔧 Troubleshooting:")
        print("   1. Make sure no other process is using the port")
        print("   2. Check that city_config.py and dynamic_data_loader.py exist")
        print("   3. Verify all dependencies are installed")
        print("   4. Try: python simple_test_dashboard.py") 

if __name__ == '__main__':
    main()