---
trigger: always_on
globs: ["**/*"]
---

# Git Branch Rules (ブランチ命名規約)

**Activation:** This rule is **ALWAYS ON** for all git branch-related operations.

## 命名形式

```
{type}/{short-description}
```

## ルール

| 項目 | 規約 | 例 |
|------|------|-----|
| **プレフィックス** | Conventional Commits の type に合わせる | `feat/`, `fix/`, `docs/`, `refactor/`, `chore/` |
| **説明部分** | 英語・小文字・ハイフン区切り | `add-search-function` |
| **文字数** | 全体で30文字以内を推奨 | `feat/add-rss-feed` |
| **スコープ** | 必要に応じてプレフィックス直後に含める | `feat/auth-add-oauth2` |

## Type 一覧（Conventional Commits 準拠）

`feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`, `build`, `ci`, `revert`

## 禁止事項

- ❌ 日本語の使用
- ❌ アンダースコア（`_`）やキャメルケース
- ❌ 大文字の使用
- ❌ 曖昧な名前: `update`, `fix-bug`, `new-feature`, `temp`, `wip`
- ❌ ブランチ名末尾のハイフンやスラッシュ

## ベースブランチ

- 分岐元は常に `v4` とする

## 例

```
feat/add-search-function
fix/header-alignment-issue
docs/update-api-reference
refactor/simplify-auth-logic
chore/upgrade-dependencies
style/unify-heading-font
perf/optimize-image-loading
```
