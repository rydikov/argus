import asyncio
import smtplib
from email.message import EmailMessage


DEFAULT_SUBJECT = 'Argus object detected'
DEFAULT_USE_TLS = True
DEFAULT_USE_SSL = False
DEFAULT_ATTACHMENT_FILENAME = 'detected.jpg'


class EmailService:
    def __init__(
        self,
        smtp_host,
        smtp_port,
        from_addr,
        to_addrs,
        username=None,
        password=None,
        use_tls=DEFAULT_USE_TLS,
        use_ssl=DEFAULT_USE_SSL,
        subject=DEFAULT_SUBJECT,
    ):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_addr = from_addr
        self.to_addrs = to_addrs
        self.use_tls = use_tls
        self.use_ssl = use_ssl
        self.subject = subject

    def _build_message(self, body):
        message = EmailMessage()
        message['Subject'] = self.subject
        message['From'] = self.from_addr
        message['To'] = ', '.join(self.to_addrs)
        message.set_content(body)
        return message

    def _build_frame_message(self, frame_bytes, message):
        email_message = self._build_message(message)
        email_message.add_attachment(
            frame_bytes,
            maintype='image',
            subtype='jpeg',
            filename=DEFAULT_ATTACHMENT_FILENAME,
        )
        return email_message

    def _send_message(self, message):
        smtp_class = smtplib.SMTP_SSL if self.use_ssl else smtplib.SMTP
        with smtp_class(self.smtp_host, self.smtp_port) as smtp:
            if self.use_tls and not self.use_ssl:
                smtp.starttls()

            if self.username and self.password:
                smtp.login(self.username, self.password)

            smtp.send_message(message, from_addr=self.from_addr, to_addrs=self.to_addrs)

    async def send_message(self, message):
        email_message = self._build_message(message)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._send_message, email_message)

    async def send_frame(self, frame, message):
        import cv2

        _, img = cv2.imencode('.jpg', frame)
        email_message = self._build_frame_message(img.tobytes(), message)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._send_message, email_message)
