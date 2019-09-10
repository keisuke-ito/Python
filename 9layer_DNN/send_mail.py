import smtplib
from email.mime.text import MIMEText

def SEND_MAIL(co_msg, subject):
    SMTP_SERVER = "smtp.mail.yahoo.co.jp"
    SMTP_PORT = 587
    SMTP_USERNAME = 'qaplokwsij@yahoo.co.jp'
    SMTP_PASSWORD = 'Ki12345678!'
    EMAIL_FROM = 'qaplokwsij@yahoo.co.jp'
    EMAIL_TO = 'oshirase_tsuti@yahoo.co.jp'

    msg = MIMEText(co_msg)
    msg['Subject'] = subject
    msg['FROM'] = EMAIL_FROM
    msg['To'] = EMAIL_TO

    mail = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
    mail.login(SMTP_USERNAME, SMTP_PASSWORD)
    mail.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())
    mail.quit()
    return

if __name__ == '__main__':
    pass
    # subject = 'About main_9l.py'
    # co_msg = 'TEST'
    # SEND_MAIL(co_msg, subject)




