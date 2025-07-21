"""
API Server Module
=================

RESTful API server for the football prediction system.
Provides endpoints for predictions, data access, and system management.
"""

from fastapi import FastAPI, HTTPException, Depends, Query, Body, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
from datetime import datetime, timedelta, date
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field, validator
import logging
import secrets
from enum import Enum

from config import Config
from database_manager import DatabaseManager
from predictor_model import FootballPredictor
from data_collector import DataCollector
from scheduler import TaskScheduler
from model_evaluator import ModelEvaluator
from backtesting import Backtester, BacktestConfig, BetType
from bet_optimizer import BetOptimizer
from cache_manager import CacheManager
from football_api import FootballAPI
from logger_module import setup_logger

logger = setup_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Football Prediction API",
    description="RESTful API for football match predictions and betting insights",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on your needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBasic()

# Global instances
config = Config()
db_manager = DatabaseManager(config)
predictor = FootballPredictor(config)
cache_manager = CacheManager(config)
football_api = FootballAPI(config)
bet_optimizer = BetOptimizer(config)


# Pydantic models for request/response
class PredictionRequest(BaseModel):
    """Match prediction request model."""
    match_id: Optional[str] = None
    home_team: Optional[str] = None
    away_team: Optional[str] = None
    league: Optional[str] = None
    date: Optional[datetime] = None
    include_live_data: bool = False


class BettingStrategyRequest(BaseModel):
    """Betting strategy optimization request."""
    capital: float = Field(gt=0, le=1000000)
    strategy: str = Field(default="kelly")
    risk_level: str = Field(default="medium")
    leagues: List[str] = Field(default_factory=list)
    min_odds: float = Field(default=1.5, ge=1.0)
    max_odds: float = Field(default=10.0, le=100.0)
    
    @validator('strategy')
    def validate_strategy(cls, v):
        valid_strategies = ['kelly', 'flat', 'value', 'mixed']
        if v not in valid_strategies:
            raise ValueError(f"Strategy must be one of {valid_strategies}")
        return v


class BacktestRequest(BaseModel):
    """Backtest request model."""
    strategy: str
    start_date: date
    end_date: date
    initial_capital: float = 1000.0
    leagues: List[str] = Field(default_factory=list)
    bet_types: List[str] = Field(default_factory=lambda: ["match_result"])
    commission: float = Field(default=0.05, ge=0, le=0.2)
    
    @validator('end_date')
    def validate_dates(cls, v, values):
        if 'start_date' in values and v <= values['start_date']:
            raise ValueError("End date must be after start date")
        return v


class SystemStatusResponse(BaseModel):
    """System status response model."""
    status: str
    uptime: str
    last_data_update: Optional[datetime]
    active_predictions: int
    model_accuracy: float
    api_version: str


# Authentication
def authenticate(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    """Authenticate API requests."""
    correct_username = secrets.compare_digest(credentials.username, config.API_USERNAME)
    correct_password = secrets.compare_digest(credentials.password, config.API_PASSWORD)
    
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    
    return credentials.username


# API Endpoints
@app.get("/", tags=["General"])
async def root():
    """API root endpoint."""
    return {
        "message": "Football Prediction API",
        "version": "1.0.0",
        "documentation": "/docs"
    }


@app.get("/health", tags=["General"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now()
    }


@app.get("/status", response_model=SystemStatusResponse, tags=["General"])
async def system_status(username: str = Depends(authenticate)):
    """Get system status and statistics."""
    try:
        # Get various system metrics
        last_update = db_manager.get_last_update_time()
        prediction_count = db_manager.count_active_predictions()
        model_metrics = cache_manager.get("latest_model_accuracy", 0.0)
        
        return SystemStatusResponse(
            status="operational",
            uptime="N/A",  # Would need to track start time
            last_data_update=last_update,
            active_predictions=prediction_count,
            model_accuracy=model_metrics,
            api_version="1.0.0"
        )
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Prediction endpoints
@app.post("/predict/match", tags=["Predictions"])
async def predict_match(
    request: PredictionRequest,
    username: str = Depends(authenticate)
):
    """Generate prediction for a specific match."""
    try:
        # Get match data
        if request.match_id:
            match = db_manager.get_match(request.match_id)
            if not match:
                raise HTTPException(status_code=404, detail="Match not found")
        else:
            # Find match by teams and date
            match = db_manager.find_match(
                home_team=request.home_team,
                away_team=request.away_team,
                date=request.date
            )
            if not match:
                raise HTTPException(status_code=404, detail="Match not found")
                
        # Check cache first
        cache_key = f"prediction:{match['id']}"
        cached_prediction = cache_manager.get(cache_key)
        
        if cached_prediction and not request.include_live_data:
            return cached_prediction
            
        # Generate prediction
        prediction = predictor.predict_match(match)
        
        # Include live data if requested and available
        if request.include_live_data and match['status'] == 'live':
            live_data = cache_manager.get(f"live:{match['id']}")
            if live_data:
                prediction = predictor.predict_live(match, live_data)
                
        # Cache prediction
        cache_manager.set(cache_key, prediction, ttl=300)  # 5 minutes
        
        # Store in database
        db_manager.store_prediction(prediction)
        
        return prediction
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating prediction: {e}")
        raise HTTPException(status_code=500, detail="Prediction generation failed")


@app.get("/predict/today", tags=["Predictions"])
async def predict_today_matches(
    league: Optional[str] = None,
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
    username: str = Depends(authenticate)
):
    """Get predictions for today's matches."""
    try:
        # Get today's matches
        matches = db_manager.get_matches_by_date(
            target_date=datetime.now().date(),
            league=league
        )
        
        predictions = []
        for match in matches:
            try:
                pred = predictor.predict_match(match)
                
                # Filter by confidence if specified
                if pred['confidence'] >= min_confidence:
                    predictions.append(pred)
                    
            except Exception as e:
                logger.warning(f"Failed to predict match {match['id']}: {e}")
                
        # Sort by confidence
        predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        return {
            "date": datetime.now().date(),
            "total_matches": len(matches),
            "predictions": predictions
        }
        
    except Exception as e:
        logger.error(f"Error getting today's predictions: {e}")
        raise HTTPException(status_code=500, detail="Failed to get predictions")


@app.get("/predict/upcoming", tags=["Predictions"])
async def predict_upcoming_matches(
    days: int = Query(7, ge=1, le=30),
    league: Optional[str] = None,
    limit: int = Query(50, ge=1, le=200),
    username: str = Depends(authenticate)
):
    """Get predictions for upcoming matches."""
    try:
        # Get upcoming matches
        end_date = datetime.now() + timedelta(days=days)
        matches = db_manager.get_upcoming_matches(
            end_date=end_date,
            league=league,
            limit=limit
        )
        
        predictions = []
        for match in matches:
            try:
                pred = predictor.predict_match(match)
                predictions.append(pred)
            except Exception as e:
                logger.warning(f"Failed to predict match {match['id']}: {e}")
                
        return {
            "period": f"Next {days} days",
            "total_matches": len(predictions),
            "predictions": predictions
        }
        
    except Exception as e:
        logger.error(f"Error getting upcoming predictions: {e}")
        raise HTTPException(status_code=500, detail="Failed to get predictions")


# Betting endpoints
@app.post("/betting/optimize", tags=["Betting"])
async def optimize_betting_strategy(
    request: BettingStrategyRequest,
    username: str = Depends(authenticate)
):
    """Optimize betting strategy for given parameters."""
    try:
        # Get upcoming matches with predictions
        matches = db_manager.get_upcoming_matches_with_predictions(
            leagues=request.leagues if request.leagues else None
        )
        
        # Filter by odds range
        filtered_matches = []
        for match in matches:
            if request.min_odds <= match['best_odds'] <= request.max_odds:
                filtered_matches.append(match)
                
        # Optimize bets
        optimization_result = bet_optimizer.optimize_portfolio(
            predictions=filtered_matches,
            total_capital=request.capital,
            strategy=request.strategy,
            risk_level=request.risk_level
        )
        
        return optimization_result
        
    except Exception as e:
        logger.error(f"Error optimizing betting strategy: {e}")
        raise HTTPException(status_code=500, detail="Strategy optimization failed")


@app.get("/betting/value-bets", tags=["Betting"])
async def get_value_bets(
    min_value: float = Query(0.05, ge=0.0, le=1.0),
    min_confidence: float = Query(0.6, ge=0.0, le=1.0),
    limit: int = Query(20, ge=1, le=100),
    username: str = Depends(authenticate)
):
    """Get current value betting opportunities."""
    try:
        # Get value bets from database
        value_bets = db_manager.get_value_bets(
            min_value=min_value,
            min_confidence=min_confidence,
            limit=limit
        )
        
        # Enhance with current odds
        for bet in value_bets:
            current_odds = cache_manager.get(f"odds:{bet['match_id']}")
            if current_odds:
                bet['current_odds'] = current_odds
                
        return {
            "timestamp": datetime.now(),
            "criteria": {
                "min_value": min_value,
                "min_confidence": min_confidence
            },
            "bets": value_bets
        }
        
    except Exception as e:
        logger.error(f"Error getting value bets: {e}")
        raise HTTPException(status_code=500, detail="Failed to get value bets")


@app.get("/betting/arbitrage", tags=["Betting"])
async def get_arbitrage_opportunities(
    username: str = Depends(authenticate)
):
    """Get current arbitrage betting opportunities."""
    try:
        # Get arbitrage opportunities
        arb_opps = db_manager.get_arbitrage_opportunities()
        
        # Validate opportunities are still valid
        valid_opps = []
        for opp in arb_opps:
            # Check if odds are still available
            current_odds = cache_manager.get(f"odds:{opp['match_id']}")
            if current_odds:
                # Recalculate arbitrage
                if bet_optimizer.calculate_arbitrage(current_odds):
                    opp['current_odds'] = current_odds
                    valid_opps.append(opp)
                    
        return {
            "timestamp": datetime.now(),
            "opportunities": valid_opps
        }
        
    except Exception as e:
        logger.error(f"Error getting arbitrage opportunities: {e}")
        raise HTTPException(status_code=500, detail="Failed to get arbitrage opportunities")


# Data endpoints
@app.get("/data/leagues", tags=["Data"])
async def get_leagues(username: str = Depends(authenticate)):
    """Get list of supported leagues."""
    try:
        leagues = db_manager.get_leagues()
        return {"leagues": leagues}
    except Exception as e:
        logger.error(f"Error getting leagues: {e}")
        raise HTTPException(status_code=500, detail="Failed to get leagues")


@app.get("/data/teams/{league_id}", tags=["Data"])
async def get_teams(
    league_id: str,
    username: str = Depends(authenticate)
):
    """Get teams in a specific league."""
    try:
        teams = db_manager.get_teams_by_league(league_id)
        return {
            "league": league_id,
            "teams": teams
        }
    except Exception as e:
        logger.error(f"Error getting teams: {e}")
        raise HTTPException(status_code=500, detail="Failed to get teams")


@app.get("/data/matches", tags=["Data"])
async def get_matches(
    league: Optional[str] = None,
    team: Optional[str] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    status: Optional[str] = None,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    username: str = Depends(authenticate)
):
    """Get matches with filters."""
    try:
        matches = db_manager.get_matches(
            league=league,
            team=team,
            start_date=start_date,
            end_date=end_date,
            status=status,
            limit=limit,
            offset=offset
        )
        
        return {
            "filters": {
                "league": league,
                "team": team,
                "start_date": start_date,
                "end_date": end_date,
                "status": status
            },
            "total": len(matches),
            "matches": matches
        }
        
    except Exception as e:
        logger.error(f"Error getting matches: {e}")
        raise HTTPException(status_code=500, detail="Failed to get matches")


@app.get("/data/match/{match_id}", tags=["Data"])
async def get_match_details(
    match_id: str,
    include_stats: bool = True,
    include_odds: bool = True,
    username: str = Depends(authenticate)
):
    """Get detailed match information."""
    try:
        match = db_manager.get_match(match_id)
        if not match:
            raise HTTPException(status_code=404, detail="Match not found")
            
        # Add statistics if requested
        if include_stats and match['status'] == 'finished':
            match['statistics'] = db_manager.get_match_statistics(match_id)
            
        # Add odds history if requested
        if include_odds:
            match['odds_history'] = db_manager.get_odds_history(match_id)
            
        return match
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting match details: {e}")
        raise HTTPException(status_code=500, detail="Failed to get match details")


@app.get("/data/standings/{league_id}", tags=["Data"])
async def get_standings(
    league_id: str,
    username: str = Depends(authenticate)
):
    """Get current league standings."""
    try:
        standings = db_manager.get_standings(league_id)
        if not standings:
            raise HTTPException(status_code=404, detail="Standings not found")
            
        return {
            "league": league_id,
            "updated": standings.get('updated'),
            "standings": standings.get('table')
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting standings: {e}")
        raise HTTPException(status_code=500, detail="Failed to get standings")


# Analytics endpoints
@app.get("/analytics/performance", tags=["Analytics"])
async def get_performance_metrics(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    league: Optional[str] = None,
    username: str = Depends(authenticate)
):
    """Get model performance metrics."""
    try:
        if not start_date:
            start_date = datetime.now().date() - timedelta(days=30)
        if not end_date:
            end_date = datetime.now().date()
            
        metrics = db_manager.get_performance_metrics(
            start_date=start_date,
            end_date=end_date,
            league=league
        )
        
        return {
            "period": {
                "start": start_date,
                "end": end_date
            },
            "metrics": metrics
        }
        
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get performance metrics")


@app.get("/analytics/roi", tags=["Analytics"])
async def get_roi_analysis(
    strategy: Optional[str] = None,
    league: Optional[str] = None,
    period_days: int = Query(30, ge=1, le=365),
    username: str = Depends(authenticate)
):
    """Get ROI analysis for betting strategies."""
    try:
        start_date = datetime.now() - timedelta(days=period_days)
        
        roi_data = db_manager.get_roi_analysis(
            start_date=start_date,
            strategy=strategy,
            league=league
        )
        
        return {
            "period_days": period_days,
            "strategy": strategy,
            "analysis": roi_data
        }
        
    except Exception as e:
        logger.error(f"Error getting ROI analysis: {e}")
        raise HTTPException(status_code=500, detail="Failed to get ROI analysis")


@app.post("/analytics/backtest", tags=["Analytics"])
async def run_backtest(
    request: BacktestRequest,
    username: str = Depends(authenticate)
):
    """Run backtest simulation."""
    try:
        # Create backtest configuration
        backtest_config = BacktestConfig(
            start_date=datetime.combine(request.start_date, datetime.min.time()),
            end_date=datetime.combine(request.end_date, datetime.min.time()),
            initial_capital=request.initial_capital,
            leagues=request.leagues,
            bet_types=[BetType(bt) for bt in request.bet_types],
            commission=request.commission
        )
        
        # Get strategy function
        strategy_map = {
            'flat': 'flat_betting_strategy',
            'kelly': 'kelly_betting_strategy',
            'value': 'value_betting_strategy'
        }
        
        if request.strategy not in strategy_map:
            raise HTTPException(status_code=400, detail="Invalid strategy")
            
        # Run backtest
        backtester = Backtester(config)
        
        # Import strategy function dynamically
        from backtesting import flat_betting_strategy, kelly_betting_strategy, value_betting_strategy
        strategy_func = locals()[strategy_map[request.strategy]]
        
        result = backtester.backtest_strategy(
            strategy_func=strategy_func,
            backtest_config=backtest_config
        )
        
        return result.to_dict()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        raise HTTPException(status_code=500, detail="Backtest failed")


# Model management endpoints
@app.post("/models/evaluate", tags=["Models"])
async def evaluate_models(username: str = Depends(authenticate)):
    """Trigger model evaluation."""
    try:
        evaluator = ModelEvaluator(config)
        results = await evaluator.evaluate_all_models()
        
        return {
            "timestamp": datetime.now(),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error evaluating models: {e}")
        raise HTTPException(status_code=500, detail="Model evaluation failed")


@app.post("/models/train", tags=["Models"])
async def train_models(
    model_type: Optional[str] = None,
    username: str = Depends(authenticate)
):
    """Trigger model training."""
    try:
        # This would typically be an async task
        # For now, return a task ID
        task_id = f"train_{datetime.now().timestamp()}"
        
        # Queue training task
        # task_queue.add_task(train_task, model_type)
        
        return {
            "task_id": task_id,
            "status": "queued",
            "message": "Model training has been queued"
        }
        
    except Exception as e:
        logger.error(f"Error queuing model training: {e}")
        raise HTTPException(status_code=500, detail="Failed to queue training")


@app.get("/models/info", tags=["Models"])
async def get_model_info(username: str = Depends(authenticate)):
    """Get information about current models."""
    try:
        model_info = {
            "match_result": {
                "last_trained": db_manager.get_model_last_trained("match_result"),
                "accuracy": cache_manager.get("model_accuracy_match_result", 0.0),
                "features": predictor.get_feature_names("match_result")
            },
            "goals": {
                "last_trained": db_manager.get_model_last_trained("goals"),
                "mae": cache_manager.get("model_mae_goals", 0.0),
                "features": predictor.get_feature_names("goals")
            }
        }
        
        return model_info
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail="Failed to get model info")


# System management endpoints
@app.post("/system/collect-data", tags=["System"])
async def trigger_data_collection(
    leagues: Optional[List[str]] = None,
    days_back: int = Query(7, ge=1, le=30),
    username: str = Depends(authenticate)
):
    """Trigger data collection."""
    try:
        collector = DataCollector(db_manager, football_api, cache_manager, config)
        
        # Run collection asynchronously
        summary = await collector.collect_all_data(
            leagues=leagues,
            days_back=days_back
        )
        
        return summary
        
    except Exception as e:
        logger.error(f"Error collecting data: {e}")
        raise HTTPException(status_code=500, detail="Data collection failed")


@app.get("/system/tasks", tags=["System"])
async def get_scheduled_tasks(username: str = Depends(authenticate)):
    """Get status of scheduled tasks."""
    try:
        scheduler = TaskScheduler(config)
        task_status = scheduler.get_task_status()
        
        return {
            "tasks": task_status,
            "system_status": scheduler.get_system_status()
        }
        
    except Exception as e:
        logger.error(f"Error getting task status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get task status")


@app.post("/system/cache/clear", tags=["System"])
async def clear_cache(
    pattern: Optional[str] = None,
    username: str = Depends(authenticate)
):
    """Clear cache entries."""
    try:
        if pattern:
            cleared = cache_manager.clear_pattern(pattern)
        else:
            cleared = cache_manager.clear_all()
            
        return {
            "cleared": cleared,
            "message": f"Cleared {cleared} cache entries"
        }
        
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear cache")


# Webhook endpoints
@app.post("/webhooks/odds-update", tags=["Webhooks"])
async def receive_odds_update(
    data: Dict[str, Any] = Body(...),
    webhook_secret: str = Query(...)
):
    """Receive odds updates from external sources."""
    try:
        # Verify webhook secret
        if webhook_secret != config.WEBHOOK_SECRET:
            raise HTTPException(status_code=403, detail="Invalid webhook secret")
            
        # Process odds update
        match_id = data.get('match_id')
        odds = data.get('odds')
        
        if match_id and odds:
            # Update cache
            cache_manager.set(f"odds:{match_id}", odds, ttl=300)
            
            # Store in database
            db_manager.insert_odds(match_id, odds)
            
            return {"status": "success", "message": "Odds updated"}
        else:
            raise HTTPException(status_code=400, detail="Invalid odds data")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing odds update: {e}")
        raise HTTPException(status_code=500, detail="Failed to process odds update")


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.now().isoformat()
        }
    )


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting Football Prediction API Server")
    
    # Initialize database
    db_manager.init_db()
    
    # Load models
    try:
        predictor.load_models()
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load models: {e}")
        
    # Start background tasks
    # scheduler.start()
    
    logger.info("API Server started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down API Server")
    
    # Stop scheduler
    # scheduler.stop()
    
    # Close database connections
    db_manager.close()
    
    logger.info("API Server shutdown complete")


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the API server."""
    uvicorn.run(
        "api_server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    # Run server
    run_server(reload=True)