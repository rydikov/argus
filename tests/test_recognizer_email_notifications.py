import importlib
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch


def import_recognizer():
    sys.modules.pop('argus.services.recognizer', None)

    openvino = types.ModuleType('openvino')
    runtime = types.ModuleType('openvino.runtime')
    runtime.Core = object
    runtime.AsyncInferQueue = object

    cv2 = types.ModuleType('cv2')
    numpy = types.ModuleType('numpy')

    modules = {
        'openvino': openvino,
        'openvino.runtime': runtime,
        'cv2': cv2,
        'numpy': numpy,
    }
    with patch.dict(sys.modules, modules):
        return importlib.import_module('argus.services.recognizer')


class RecognizerEmailNotificationsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recognizer_module = import_recognizer()

    def make_recognizer(self):
        recognizer = self.recognizer_module.OpenVinoRecognizer.__new__(
            self.recognizer_module.OpenVinoRecognizer
        )
        recognizer.telegram = Mock()
        recognizer.email_service = Mock()
        return recognizer

    def test_confirmed_detection_sends_link_to_telegram_and_email(self):
        recognizer = self.make_recognizer()
        throttler = Mock()
        throttler.is_allowed.return_value = True
        queue_item = SimpleNamespace(
            thread_name='first-cam',
            url='http://example.com/Stills/detected.jpg',
            frame=object(),
            object_detected_prompt='Object detected.',
        )

        with patch.dict(self.recognizer_module.notification_throttlers, {'first-cam': throttler}, clear=True):
            with patch.object(self.recognizer_module, 'run_async') as run_async:
                recognizer.notify_on_confirmed_detection(queue_item, detection_is_confirm=True)

        throttler.is_allowed.assert_called_once_with()
        self.assertEqual(run_async.call_count, 2)
        run_async.assert_any_call(
            recognizer.telegram.send_message,
            'Objects detected: http://example.com/Stills/detected.jpg',
        )
        run_async.assert_any_call(
            recognizer.email_service.send_message,
            'Objects detected: http://example.com/Stills/detected.jpg',
        )

    def test_confirmed_detection_without_url_sends_frame_to_email(self):
        recognizer = self.make_recognizer()
        throttler = Mock()
        throttler.is_allowed.return_value = True
        frame = object()
        queue_item = SimpleNamespace(
            thread_name='first-cam',
            url=None,
            frame=frame,
            object_detected_prompt='Object detected.',
        )

        with patch.dict(self.recognizer_module.notification_throttlers, {'first-cam': throttler}, clear=True):
            with patch.object(self.recognizer_module, 'run_async') as run_async:
                recognizer.notify_on_confirmed_detection(queue_item, detection_is_confirm=True)

        run_async.assert_any_call(
            recognizer.telegram.send_frame,
            frame,
            'Object detected.',
        )
        run_async.assert_any_call(
            recognizer.email_service.send_frame,
            frame,
            'Object detected.',
        )

    def test_unconfirmed_detection_does_not_send_email(self):
        recognizer = self.make_recognizer()
        throttler = Mock()
        queue_item = SimpleNamespace(
            thread_name='first-cam',
            url='http://example.com/Stills/detected.jpg',
            frame=object(),
            object_detected_prompt='Object detected.',
        )

        with patch.dict(self.recognizer_module.notification_throttlers, {'first-cam': throttler}, clear=True):
            with patch.object(self.recognizer_module, 'run_async') as run_async:
                recognizer.notify_on_confirmed_detection(queue_item, detection_is_confirm=False)

        throttler.is_allowed.assert_not_called()
        run_async.assert_not_called()


if __name__ == '__main__':
    unittest.main()
