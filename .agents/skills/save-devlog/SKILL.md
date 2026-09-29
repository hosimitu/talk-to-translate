---
name: save-devlog
description: 開発作業、機能実装、バグ修正、リファクタリングの完了時に、作業ログをObsidianの適切なディレクトリへ構造化Markdownとして記録する。
---

# 開発ログ記録ルール

## 1. ファイル命名規則
`YYYY-MM-DD_タスク名の短縮形.md`
（例: `2026-09-27_auth-refactoring.md`）

## 実行手順
1. タスク完了時に出力された「Implementation Plan」および「Walkthrough」から要点を抽出する。
2. ターミナルツールを用いて直近のGitコミットハッシュ（git rev-parse --short HEAD）を取得する。
3. 以下のYAMLフロントマターを持つMarkdownファイルを生成する。
4. 作業が完了した際、以下のフォーマットでMarkdownファイルを生成し、Obsidian保管庫の `DevLogs/talk-to-translate/` 配下に保存してください。
5. `DevLogs/talk-to-translate/` 配下にindex.mdを作成し、生成されたMarkdownファイルのリンクを追加することで、Obsidian内で容易に参照できるようにしてください。

## 3. 出力フォーマット
以下のYAMLフロントマターおよび見出し構成を厳密に維持して出力してください。

```markdown
---
type: devlog
date: {{YYYY-MM-DD}}
project: {{プロジェクト名}}
commit: {{最新のGitコミットハッシュ（取得可能な場合）}}
status: completed
tags:
  - devlog
  - {{関連技術スタック}}
---

# {{タスクまたは実装作業のタイトル}}

## 1. 概要と背景
- {{作業の目的や解決した課題}}

## 2. 実施した変更
- {{修正したファイルや追加した主要ロジックの要約}}

## 3. 検証・テスト
- 実行コマンド: `{{テストコマンド}}`
- 検証結果: {{正常動作の確認内容}}

## 4. 参照リンク
- 関連ノート: [[{{関連する仕様書や既存ノート名}}]]
```