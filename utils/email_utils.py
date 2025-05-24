# utils/email_utils.py
"""
Utility functions for sending emails.
This is a placeholder. Actual email sending requires integration with an
SMTP server or a transactional email service (e.g., SendGrid, Mailgun).
Credentials for such services should be stored securely, e.g., via Streamlit secrets.
"""
import logging
import streamlit as st # For accessing secrets in a deployed app

try:
    from config import APP_TITLE, EMAIL_SENDER_ADDRESS # For logger and default sender
except ImportError:
    APP_TITLE = "TradingDashboard_Default"
    EMAIL_SENDER_ADDRESS = "noreply@example.com" # Fallback

logger = logging.getLogger(APP_TITLE)

# --- Placeholder for Email Configuration (to be set via Streamlit secrets or .env) ---
# Example for SendGrid (replace with your actual provider's setup)
# SENDGRID_API_KEY = st.secrets.get("sendgrid_api_key")
# SMTP_SERVER = st.secrets.get("smtp_server")
# SMTP_PORT = st.secrets.get("smtp_port", 587)
# SMTP_USERNAME = st.secrets.get("smtp_username")
# SMTP_PASSWORD = st.secrets.get("smtp_password")


def send_password_reset_email(recipient_email: str, user_name: str, reset_link: str) -> bool:
    """
    Sends a password reset email to the user.
    This is a placeholder and will only log the action.

    Args:
        recipient_email (str): The email address of the user.
        user_name (str): The username of the user.
        reset_link (str): The unique link for resetting the password.

    Returns:
        bool: True if the email was "sent" (logged) successfully, False otherwise.
    """
    subject = f"Password Reset Request for {APP_TITLE}"
    body_text = f"""
    Hi {user_name},

    We received a request to reset your password for your {APP_TITLE} account.
    If you did not request this, please ignore this email.

    To reset your password, please click on the link below or copy and paste it into your browser:
    {reset_link}

    This link will expire in 1 hour.

    Thanks,
    The {APP_TITLE} Team
    """
    body_html = f"""
    <html>
        <body>
            <p>Hi {user_name},</p>
            <p>We received a request to reset your password for your {APP_TITLE} account.<br>
            If you did not request this, please ignore this email.</p>
            <p>To reset your password, please click on the link below or copy and paste it into your browser:</p>
            <p><a href="{reset_link}">{reset_link}</a></p>
            <p>This link will expire in <strong>1 hour</strong>.</p>
            <p>Thanks,<br>The {APP_TITLE} Team</p>
        </body>
    </html>
    """

    logger.info(f"--- SIMULATING PASSWORD RESET EMAIL ---")
    logger.info(f"To: {recipient_email}")
    logger.info(f"From: {EMAIL_SENDER_ADDRESS}")
    logger.info(f"Subject: {subject}")
    logger.info(f"Body (Text - for logs):\n{body_text}")
    logger.info(f"Reset Link (for testing): {reset_link}")
    logger.info(f"--- END OF SIMULATED EMAIL ---")

    # In a real implementation, you would use a library like `sendgrid`, `smtplib`, etc.
    # Example using smtplib (requires SMTP server setup):
    # import smtplib
    # from email.mime.text import MIMEText
    # from email.mime.multipart import MIMEMultipart
    #
    # if not SMTP_SERVER or not SMTP_USERNAME or not SMTP_PASSWORD:
    #     logger.error("SMTP configuration is missing. Cannot send actual email.")
    #     st.error("Email server not configured. Password reset email cannot be sent.")
    #     return False
    #
    # try:
    #     msg = MIMEMultipart('alternative')
    #     msg['Subject'] = subject
    #     msg['From'] = EMAIL_SENDER_ADDRESS
    #     msg['To'] = recipient_email
    #
    #     part1 = MIMEText(body_text, 'plain')
    #     part2 = MIMEText(body_html, 'html')
    #     msg.attach(part1)
    #     msg.attach(part2)
    #
    #     with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
    #         server.starttls() # Use TLS
    #         server.login(SMTP_USERNAME, SMTP_PASSWORD)
    #         server.sendmail(EMAIL_SENDER_ADDRESS, recipient_email, msg.as_string())
    #     logger.info(f"Password reset email successfully sent to {recipient_email} (actual send).")
    #     return True
    # except Exception as e:
    #     logger.error(f"Failed to send password reset email to {recipient_email}: {e}", exc_info=True)
    #     st.error(f"Could not send password reset email. Please try again later or contact support. Error: {e}")
    #     return False

    # For this placeholder, we always return True after logging.
    st.success(f"A password reset link would have been sent to {recipient_email} if email services were configured. Please check the application logs for the link (for testing purposes).")
    return True

if __name__ == "__main__":
    # Test the placeholder function
    print("Testing send_password_reset_email placeholder...")
    test_email = "testuser@example.com"
    test_username = "testuser"
    test_reset_url = "http://localhost:8501/?page=reset_password&token=TEST_TOKEN_12345" # Example
    
    # To make st.secrets work here if running directly, you'd need a .streamlit/secrets.toml
    # For now, the placeholder doesn't rely on it.
    
    if send_password_reset_email(test_email, test_username, test_reset_url):
        print(f"Placeholder email 'sent' to {test_email}. Check logs.")
    else:
        print(f"Placeholder email 'sending' failed for {test_email}.")

