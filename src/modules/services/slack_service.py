from __future__ import annotations

import os
from typing import Any

import requests
from dotenv import load_dotenv

from src.modules.helper.helper import Helper

load_dotenv(dotenv_path=Helper.ROOT / ".env", override=False)


class SlackService:
    """Slack への通知を行うサービスクラス。

    - Webhook URL は環境変数 `SLACK_WEBHOOK_URL` を利用。
    - 必要に応じてプロジェクトルートの `.env` を `python-dotenv` で読み込む。
    - 送信は Incoming Webhook を想定し、`{"text": "..."}` を POST する。
    """

    def __init__(self) -> None:
        """`.env` 読み込み後に Webhook URL を確定する。"""

        webhook_url = os.getenv("SLACK_WEBHOOK_URL")
        if not webhook_url:
            raise RuntimeError("SLACK_WEBHOOK_URL が設定されていません (.env または環境変数)。")

        self._webhook_url: str = webhook_url

    def send(self, text: str) -> bool:
        """テキストメッセージを Slack に送信する。"""
        payload: dict[str, Any] = {"text": text}
        res = requests.post(self._webhook_url, json=payload, timeout=10)
        return res.ok
