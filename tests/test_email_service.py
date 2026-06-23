import asyncio
import unittest
from unittest.mock import patch

from argus.services.email import EmailService


class EmailServiceTest(unittest.TestCase):
    def test_send_message_uses_tls_login_and_recipients(self):
        service = EmailService(
            smtp_host='smtp.example.com',
            smtp_port=587,
            username='user@example.com',
            password='secret',
            from_addr='argus@example.com',
            to_addrs=['alerts@example.com'],
            use_tls=True,
            use_ssl=False,
            subject='Argus alert',
        )

        with patch('argus.services.email.smtplib.SMTP') as smtp_class:
            smtp = smtp_class.return_value.__enter__.return_value

            asyncio.run(service.send_message('Objects detected: http://example.com/frame.jpg'))

        smtp_class.assert_called_once_with('smtp.example.com', 587)
        smtp.starttls.assert_called_once_with()
        smtp.login.assert_called_once_with('user@example.com', 'secret')
        smtp.send_message.assert_called_once()

        message = smtp.send_message.call_args.args[0]
        self.assertEqual(message['Subject'], 'Argus alert')
        self.assertEqual(message['From'], 'argus@example.com')
        self.assertEqual(message['To'], 'alerts@example.com')
        self.assertEqual(message.get_content().strip(), 'Objects detected: http://example.com/frame.jpg')
        self.assertEqual(smtp.send_message.call_args.kwargs['from_addr'], 'argus@example.com')
        self.assertEqual(smtp.send_message.call_args.kwargs['to_addrs'], ['alerts@example.com'])

    def test_ssl_preferred_over_starttls(self):
        service = EmailService(
            smtp_host='smtp.example.com',
            smtp_port=465,
            username=None,
            password=None,
            from_addr='argus@example.com',
            to_addrs=['alerts@example.com'],
            use_tls=True,
            use_ssl=True,
        )

        with patch('argus.services.email.smtplib.SMTP_SSL') as smtp_ssl_class:
            smtp = smtp_ssl_class.return_value.__enter__.return_value

            asyncio.run(service.send_message('Objects detected'))

        smtp_ssl_class.assert_called_once_with('smtp.example.com', 465)
        smtp.starttls.assert_not_called()
        smtp.login.assert_not_called()
        smtp.send_message.assert_called_once()

    def test_frame_message_contains_jpeg_attachment(self):
        service = EmailService(
            smtp_host='smtp.example.com',
            smtp_port=587,
            from_addr='argus@example.com',
            to_addrs=['alerts@example.com'],
        )

        message = service._build_frame_message(b'jpeg bytes', 'Object detected.')
        attachments = list(message.iter_attachments())

        self.assertEqual(message['Subject'], 'Argus object detected')
        self.assertEqual(message.get_body(preferencelist=('plain',)).get_content().strip(), 'Object detected.')
        self.assertEqual(len(attachments), 1)
        self.assertEqual(attachments[0].get_filename(), 'detected.jpg')
        self.assertEqual(attachments[0].get_content_type(), 'image/jpeg')
        self.assertEqual(attachments[0].get_content(), b'jpeg bytes')


if __name__ == '__main__':
    unittest.main()
