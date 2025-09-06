#!/usr/bin/env python3
"""
testデータセットの特徴抽出スクリプト

このスクリプトはManagerを使用してtestデータセットに対して
ESM2とESM1bモデルでの特徴抽出を実行します。
"""

import logging

from src.main.configs.extraction.test import test_config
from src.main.utils.manager import Manager


def setup_logging():
    """ログ設定を初期化"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("test_extraction.log"), logging.StreamHandler()],
    )


def main():
    """メイン処理"""
    # ログ設定
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("=== test特徴抽出スクリプト開始 ===")

    try:
        # 両方のモデル設定を使用
        configs = [test_config]
        logger.info(f"{len(configs)}個の抽出設定を準備しました")

        # Managerで実行
        manager = Manager(configs)
        logger.info("Manager実行開始")

        manager.run_all()

        # 結果を報告
        successful_count = len(manager.get_results())
        error_count = len(manager.get_errors())

        logger.info("=== 実行結果 ===")
        logger.info(f"成功: {successful_count}個")
        logger.info(f"失敗: {error_count}個")

        if manager.has_errors():
            logger.warning("以下のエラーが発生しました:")
            for i, error in enumerate(manager.get_errors(), 1):
                logger.error(f"エラー{i}: {error}")

        if successful_count > 0:
            logger.info("成功した実行結果:")
            for i, result in enumerate(manager.get_results(), 1):
                logger.info(f"結果{i}: {len(result)}個のタンパク質を処理")

        logger.info("=== test特徴抽出スクリプト完了 ===")

        return successful_count > 0

    except Exception as e:
        logger.error(f"スクリプト実行中に予期しないエラーが発生しました: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
