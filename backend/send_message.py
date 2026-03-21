"""
Firebase Cloud Messaging (FCM) Message Sender

This module handles sending push notifications to mobile devices via Firebase.
All message data values must be strings as per FCM requirements.
"""
from firebase_admin import messaging
import logging

logger = logging.getLogger(__name__)

def send_data_message(data: dict):
    """
    Send a data message to all subscribed users via Firebase Cloud Messaging.
    
    FCM data messages are used for sending structured data to mobile apps.
    All values in the data dictionary must be strings (FCM requirement).
    
    Args:
        data: Dictionary containing alert data to send. Keys and values will be
              converted to strings. Typical fields include:
              - symbol: Stock ticker symbol
              - Predicted_change: Predicted percentage change
              - News: News headlines
              - Sentiment Score: Sentiment analysis score
              - close: Current closing price
              - sigma_forecast: Volatility forecast
              - ema_filter_trend_up/down: EMA trend indicators
    
    Returns:
        str: Message ID returned by Firebase if successful
    
    Raises:
        Exception: If message sending fails (network error, invalid data, etc.)
    
    Note:
        Messages are sent to topic "all_users" - all devices subscribed to this
        topic will receive the notification.
    """
    try:
        # Convert all values to strings (FCM requirement)
        # FCM data messages only accept string key-value pairs
        data = {k: str(v) for k, v in data.items()}
        
        logger.debug(f"Preparing FCM message with {len(data)} fields")

        # Create FCM message targeting all subscribed users
        message = messaging.Message(
            topic="all_users",  # Topic subscription (all users subscribed to this topic)
            data=data  # Data payload (all values must be strings)
        )

        # Send message via Firebase Cloud Messaging
        response = messaging.send(message)
        logger.info(f"FCM message sent successfully. Message ID: {response}")
        return response
        
    except ValueError as e:
        # Invalid message format or data
        logger.error(f"Invalid FCM message format: {str(e)}", exc_info=True)
        raise
    except Exception as e:
        # Network errors, authentication errors, etc.
        logger.error(f"Failed to send FCM message: {str(e)}", exc_info=True)
        raise

