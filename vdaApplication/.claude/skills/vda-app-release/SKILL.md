---
name: vda-app-release
description: Build, sign and publish a release APK of the VDA app to a GitHub release -- signing config, the PKCS12 password trap, ABI packaging, artifact naming and checksums. Use when cutting a version, when assembleRelease produces an unsigned APK or fails with "Given final block not properly padded", or before uploading anything users will install.
---

# VDA app: building and publishing a release

Distribution is a **single APK attached to a GitHub release** (not Play), so
one universal APK must install on every supported device -- see "ABIs" below.

## Signing

Signing material is deliberately outside version control (`.gitignore` excludes
`*.keystore`, `*.jks`, `keystore.properties`, `dist/`).
`app/build.gradle.kts` reads `keystore.properties` from the project root:

```properties
storeFile=vda-release.keystore
storePassword=...
keyAlias=vda-release
keyPassword=...
```

If that file is **absent the build still succeeds**, just unsigned -- a fresh
clone is never blocked by a missing secret. If `assembleRelease` gives you an
unsigned APK, that file is missing or unreadable.

The key is 2048-bit RSA, alias `vda-release`, valid 10,000 days. Only **v3**
signing is enabled: v1/v2 are redundant at `minSdk 35`, and v3 is what
permits key rotation later. `apksigner verify -v` reports `v2: false, v3:
true` -- that is expected.

### The PKCS12 trap

`keytool` creates PKCS12 keystores by default, and **PKCS12 has no separate
key password**: `keytool -genkeypair ... -keypass X` silently uses the store
password instead. If `keystore.properties` then carries a different
`keyPassword`, packaging fails with

```
Failed to read key vda-release from store ...: Given final block not properly padded
```

Set `keyPassword` equal to `storePassword`. The comment in
`keystore.properties` says so; keep it.

> **The keystore is irreplaceable.** Android installs an update over an existing
> install only if it is signed with the same key. Lose it and every user must
> uninstall before they can update. The keystore and `keystore.properties` are
> backed up in `~/keys/vda/` on the development machine; keep that copy
> current -- a gitignored file is still destroyed by `git clean -xdf`.

## ABIs

`ndk { abiFilters += listOf("arm64-v8a", "x86_64") }`. TFLite native libs are
~70 MB per ABI; `armeabi-v7a`/`x86` are unreachable at `minSdk 35`. Keep
**both** remaining ABIs in the published APK -- `arm64-v8a` is every real
phone, `x86_64` covers emulators and Chromebooks. Do **not** narrow further for
a public release; use ABI splits or an AAB if per-device size ever matters.

The release APK is ~210 MB, all of it native TFLite libraries; there are no
model weights inside (see `vda-app-models`).

## Cutting a version

1. Bump `versionCode` (must increase every release) and `versionName` in
   `app/build.gradle.kts`.
2. Build both variants:

```bash
./gradlew assembleDebug assembleRelease
```

3. Verify the signature:

```bash
$ANDROID_HOME/build-tools/<ver>/apksigner verify -v \
  app/build/outputs/apk/release/app-release.apk        # expect: Verifies, v3 true
```

4. Stage with **release-artifact naming** -- the release APK carries **no
   `-release` suffix**; only non-default variants are qualified:

```bash
mkdir -p dist
cp app/build/outputs/apk/release/app-release.apk dist/vda-<version>.apk
cp app/build/outputs/apk/debug/app-debug.apk     dist/vda-<version>-debug.apk
sha256sum dist/*.apk
```

5. Install the *release* APK on a real device -- `pm uninstall` first, since a
   debug-signed build cannot be updated in place -- and run a full depth job
   from a fresh download before publishing (see `vda-app-verify`). Check the
   output is a real depth map, not a flat one.

6. Publish. **Tags are component-prefixed** -- this repo also publishes the
   model weights from the same release list (`models-v1`) and keeps the app
   and converter on separate branches, so a bare `v1.0` would be ambiguous.
   App releases are `app-v<version>` from the `application` branch:

```bash
gh release create app-v<version> \
  --target application \
  --title "VDA Android App v<version>" \
  --notes-file <notes>.md \
  --draft \
  dist/vda-<version>.apk

gh release download app-v<version> -p 'vda-<version>.apk' -D /tmp/verify
sha256sum /tmp/verify/vda-<version>.apk dist/vda-<version>.apk   # must match

gh release edit app-v<version> --draft=false
```

Draft first, always: the tag is not created until a draft is published, so a
bad upload can be discarded without leaving a tag behind, and the round-trip
checksum catches a truncated 200 MB transfer before users hit it. Publish the
SHA-256 in the release notes, and only then add the Download section to
`README.md` and the release link to `USER_GUIDE.md` -- both are deliberately
absent until something is published.

## Two things to get right

- **Do not publish the debug APK to users.** It is `debuggable`, signed with the
  shared Android debug key, allows a debugger to attach, and triggers the
  platform's "not 16 KB-compatible" warning dialog on every launch. Build it
  for local testing; attach only the release APK.
- **R8 is off** (`optimization { enable = false }`). Enabling it would shrink
  the APK but risks breaking TFLite's reflection-based paths, so it needs a
  real depth run to validate -- treat it as its own change, never as part of
  cutting a release.
