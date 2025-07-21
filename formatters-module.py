"""
Output formatting utilities for ProFootballAI
"""

from typing import Union, Any, List, Dict, Optional
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np
from decimal import Decimal, ROUND_HALF_UP


def format_currency(
    amount: Union[int, float, Decimal],
    currency: str = "€",
    decimal_places: int = 2,
    thousands_separator: str = ",",
    decimal_separator: str = "."
) -> str:
    """
    Format amount as currency string
    
    Args:
        amount: The amount to format
        currency: Currency symbol
        decimal_places: Number of decimal places
        thousands_separator: Separator for thousands
        decimal_separator: Decimal separator
        
    Returns:
        Formatted currency string
    """
    
    # Handle None
    if amount is None:
        return f"{currency}0{decimal_separator}00"
    
    # Convert to Decimal for precise formatting
    if not isinstance(amount, Decimal):
        amount = Decimal(str(amount))
    
    # Round to specified decimal places
    quantizer = Decimal(f"0.{'0' * decimal_places}")
    amount = amount.quantize(quantizer, rounding=ROUND_HALF_UP)
    
    # Format with separators
    parts = f"{amount:.{decimal_places}f}".split('.')
    
    # Add thousands separator
    integer_part = parts[0]
    formatted_integer = ""
    
    for i, digit in enumerate(reversed(integer_part)):
        if i > 0 and i % 3 == 0:
            formatted_integer = thousands_separator + formatted_integer
        formatted_integer = digit + formatted_integer
    
    # Combine parts
    if decimal_places > 0:
        result = f"{formatted_integer}{decimal_separator}{parts[1]}"
    else:
        result = formatted_integer
    
    # Add currency symbol
    if amount < 0:
        return f"-{currency}{result[1:]}"
    else:
        return f"{currency}{result}"


def format_percentage(
    value: Union[int, float],
    decimal_places: int = 1,
    include_sign: bool = True
) -> str:
    """
    Format value as percentage
    
    Args:
        value: Value to format (0.5 = 50%)
        decimal_places: Number of decimal places
        include_sign: Whether to include % sign
        
    Returns:
        Formatted percentage string
    """
    
    if value is None:
        return "0%" if include_sign else "0"
    
    percentage = value * 100
    
    if decimal_places == 0:
        formatted = f"{percentage:.0f}"
    else:
        formatted = f"{percentage:.{decimal_places}f}"
    
    if include_sign:
        return f"{formatted}%"
    else:
        return formatted


def format_odds(
    odds: Union[int, float],
    style: str = "decimal",
    include_prefix: bool = True
) -> str:
    """
    Format betting odds in different styles
    
    Args:
        odds: Decimal odds value
        style: 'decimal', 'fractional', or 'american'
        include_prefix: Whether to include @ prefix
        
    Returns:
        Formatted odds string
    """
    
    if odds is None or odds <= 0:
        return "N/A"
    
    prefix = "@" if include_prefix else ""
    
    if style == "decimal":
        return f"{prefix}{odds:.2f}"
        
    elif style == "fractional":
        # Convert decimal to fractional
        if odds == 1:
            return f"{prefix}0/1"
        
        numerator = int((odds - 1) * 100)
        denominator = 100
        
        # Simplify fraction
        from math import gcd
        common = gcd(numerator, denominator)
        numerator //= common
        denominator //= common
        
        return f"{prefix}{numerator}/{denominator}"
        
    elif style == "american":
        # Convert decimal to American
        if odds >= 2:
            american = int((odds - 1) * 100)
            return f"{prefix}+{american}"
        else:
            american = int(-100 / (odds - 1))
            return f"{prefix}{american}"
            
    else:
        return f"{prefix}{odds:.2f}"


def format_date(
    date_value: Union[str, datetime, date],
    format_str: str = "%Y-%m-%d",
    relative: bool = False
) -> str:
    """
    Format date with optional relative formatting
    
    Args:
        date_value: Date to format
        format_str: strftime format string
        relative: Whether to use relative formatting (Today, Tomorrow, etc.)
        
    Returns:
        Formatted date string
    """
    
    if date_value is None:
        return "N/A"
    
    # Convert to datetime if needed
    if isinstance(date_value, str):
        try:
            date_value = pd.to_datetime(date_value)
        except:
            return date_value
            
    if isinstance(date_value, date) and not isinstance(date_value, datetime):
        date_value = datetime.combine(date_value, datetime.min.time())
    
    if relative:
        today = datetime.now().date()
        target_date = date_value.date() if isinstance(date_value, datetime) else date_value
        
        diff = (target_date - today).days
        
        if diff == 0:
            return "Today"
        elif diff == 1:
            return "Tomorrow"
        elif diff == -1:
            return "Yesterday"
        elif 0 < diff <= 7:
            return f"In {diff} days"
        elif -7 <= diff < 0:
            return f"{-diff} days ago"
    
    return date_value.strftime(format_str)


def format_time(
    time_value: Union[str, datetime],
    format_str: str = "%H:%M",
    include_seconds: bool = False
) -> str:
    """Format time value"""
    
    if time_value is None:
        return "N/A"
    
    if isinstance(time_value, str):
        try:
            time_value = pd.to_datetime(time_value)
        except:
            return time_value
    
    if include_seconds:
        format_str = "%H:%M:%S"
    
    return time_value.strftime(format_str)


def format_duration(
    seconds: Union[int, float],
    style: str = "short"
) -> str:
    """
    Format duration in seconds to human readable
    
    Args:
        seconds: Duration in seconds
        style: 'short' (1h 30m) or 'long' (1 hour 30 minutes)
        
    Returns:
        Formatted duration string
    """
    
    if seconds is None or seconds < 0:
        return "N/A"
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    parts = []
    
    if style == "short":
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        if secs > 0 or not parts:
            parts.append(f"{secs}s")
    else:
        if hours > 0:
            parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
        if minutes > 0:
            parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
        if secs > 0 or not parts:
            parts.append(f"{secs} second{'s' if secs != 1 else ''}")
    
    return " ".join(parts)


def format_number(
    number: Union[int, float],
    decimal_places: int = 0,
    thousands_separator: str = ",",
    compact: bool = False
) -> str:
    """
    Format number with optional compact notation
    
    Args:
        number: Number to format
        decimal_places: Decimal places
        thousands_separator: Separator for thousands
        compact: Use compact notation (1.2K, 3.4M, etc.)
        
    Returns:
        Formatted number string
    """
    
    if number is None:
        return "0"
    
    if compact and abs(number) >= 1000:
        suffixes = ['', 'K', 'M', 'B', 'T']
        magnitude = 0
        
        while abs(number) >= 1000 and magnitude < len(suffixes) - 1:
            number /= 1000
            magnitude += 1
        
        return f"{number:.{decimal_places}f}{suffixes[magnitude]}"
    
    # Regular formatting
    if decimal_places == 0:
        formatted = f"{int(number):,}"
    else:
        formatted = f"{number:,.{decimal_places}f}"
    
    if thousands_separator != ",":
        formatted = formatted.replace(",", thousands_separator)
    
    return formatted


def format_team_name(
    team_name: str,
    max_length: int = None,
    abbreviate: bool = False
) -> str:
    """
    Format team name with optional abbreviation
    
    Args:
        team_name: Full team name
        max_length: Maximum length
        abbreviate: Whether to abbreviate
        
    Returns:
        Formatted team name
    """
    
    if not team_name:
        return "N/A"
    
    # Common abbreviations
    abbreviations = {
        "Manchester United": "Man Utd",
        "Manchester City": "Man City",
        "Tottenham Hotspur": "Tottenham",
        "West Ham United": "West Ham",
        "Newcastle United": "Newcastle",
        "Real Madrid": "Real",
        "Atletico Madrid": "Atletico",
        "Paris Saint-Germain": "PSG",
        "Bayern Munich": "Bayern",
        "Borussia Dortmund": "BVB",
        "Inter Milan": "Inter",
        "AC Milan": "Milan"
    }
    
    if abbreviate and team_name in abbreviations:
        team_name = abbreviations[team_name]
    
    if max_length and len(team_name) > max_length:
        return team_name[:max_length-3] + "..."
    
    return team_name


def format_match_result(
    home_score: int,
    away_score: int,
    include_outcome: bool = True
) -> str:
    """Format match result"""
    
    result = f"{home_score} - {away_score}"
    
    if include_outcome:
        total = home_score + away_score
        if total > 2.5:
            result += " ✅ Over 2.5"
        else:
            result += " ❌ Under 2.5"
    
    return result


def format_form(
    form: Union[str, List[str]],
    style: str = "emoji",
    last_n: int = 5
) -> str:
    """
    Format team form string
    
    Args:
        form: Form string (WWDLL) or list ['W', 'W', 'D', 'L', 'L']
        style: 'emoji', 'colored', or 'plain'
        last_n: Number of matches to show
        
    Returns:
        Formatted form string
    """
    
    if not form:
        return "N/A"
    
    # Convert to list if string
    if isinstance(form, str):
        form = list(form)
    
    # Take last N
    form = form[-last_n:]
    
    if style == "emoji":
        mapping = {'W': '✅', 'D': '➖', 'L': '❌'}
    elif style == "colored":
        mapping = {
            'W': '<span style="color: #00d4aa;">W</span>',
            'D': '<span style="color: #ffa502;">D</span>',
            'L': '<span style="color: #ff4757;">L</span>'
        }
    else:
        mapping = {'W': 'W', 'D': 'D', 'L': 'L'}
    
    formatted = [mapping.get(result, result) for result in form]
    
    if style == "colored":
        return " ".join(formatted)
    else:
        return "".join(formatted)


def format_confidence_level(
    confidence: Union[str, float],
    style: str = "text"
) -> str:
    """
    Format confidence level
    
    Args:
        confidence: Confidence level or score
        style: 'text', 'emoji', or 'bar'
        
    Returns:
        Formatted confidence string
    """
    
    # Convert score to level if needed
    if isinstance(confidence, (int, float)):
        if confidence >= 0.8:
            level = "High"
        elif confidence >= 0.6:
            level = "Medium"
        else:
            level = "Low"
    else:
        level = confidence
    
    if style == "emoji":
        mapping = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}
        return f"{mapping.get(level, '⚪')} {level}"
    elif style == "bar":
        bars = {"High": "████████", "Medium": "█████", "Low": "██"}
        return f"{bars.get(level, '█')} {level}"
    else:
        return level


def format_dataframe_display(
    df: pd.DataFrame,
    currency_columns: List[str] = None,
    percentage_columns: List[str] = None,
    date_columns: List[str] = None,
    number_columns: List[str] = None
) -> pd.DataFrame:
    """
    Format DataFrame columns for display
    
    Args:
        df: DataFrame to format
        currency_columns: Columns to format as currency
        percentage_columns: Columns to format as percentage
        date_columns: Columns to format as dates
        number_columns: Columns to format as numbers
        
    Returns:
        Formatted DataFrame
    """
    
    formatted_df = df.copy()
    
    # Format currency columns
    if currency_columns:
        for col in currency_columns:
            if col in formatted_df.columns:
                formatted_df[col] = formatted_df[col].apply(
                    lambda x: format_currency(x) if pd.notna(x) else "N/A"
                )
    
    # Format percentage columns
    if percentage_columns:
        for col in percentage_columns:
            if col in formatted_df.columns:
                formatted_df[col] = formatted_df[col].apply(
                    lambda x: format_percentage(x) if pd.notna(x) else "N/A"
                )
    
    # Format date columns
    if date_columns:
        for col in date_columns:
            if col in formatted_df.columns:
                formatted_df[col] = formatted_df[col].apply(
                    lambda x: format_date(x) if pd.notna(x) else "N/A"
                )
    
    # Format number columns
    if number_columns:
        for col in number_columns:
            if col in formatted_df.columns:
                formatted_df[col] = formatted_df[col].apply(
                    lambda x: format_number(x, compact=True) if pd.notna(x) else "N/A"
                )
    
    return formatted_df