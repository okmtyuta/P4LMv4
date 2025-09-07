import platform


class PlatformService:
    """プラットフォーム関連のユーティリティサービス。"""

    @classmethod
    def server_name(cls) -> str:
        """サーバー（ホスト）の名前を取得して返す。"""
        return platform.node()
