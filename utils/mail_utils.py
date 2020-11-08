import smtplib, ssl
from configs.global_vars import EMAIL_USER, EMAIL_PASSWORD

SMTP_MAIL = "smtp.mailtrap.io"
PORT = 2525
SENDER = "prim@inbox.mailtrap.io"
RECEIVER = "prim@inbox.mailtrap.io"
MESSAGE_TEMPLATE = "Subject: {}\nTo: {}\nFrom: {}\n\n{}"

def send_email(subject, message):
    with smtplib.SMTP(SMTP_MAIL, PORT) as server:
        server.login(EMAIL_USER, EMAIL_PASSWORD)
        content = MESSAGE_TEMPLATE.format(subject, RECEIVER, SENDER, message)
        server.sendmail(SENDER, RECEIVER, content)