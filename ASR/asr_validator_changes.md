# ASR Validator Command-Line Integration Changes

## Purpose
Modified `ASR/asr_validator.py` to support JSON output for subprocess integration with the main Pocket-TTS audiobook generator. This enables the generator to validate regenerated audio chunks during ASR quality control reprocessing.

## Changes Summary
Two modifications were made to enable command-line JSON output:
1. Added `--json` argument to the argument parser
2. Modified single-chunk validation mode to output JSON when flag is set
3. Added `--audio-path` argument to validate regenerated attempt WAV files while reusing the original text file

## Complete Change Details

### Change 1: Add --json Argument

**File:** `ASR/asr_validator.py`
**Location:** ~Line 1643 (in the main execution block argument parser)

**Original Code:**
```python
    parser.add_argument('--single-chunk', help='Validate a single chunk (e.g., chunk_00005)')

    args = parser.parse_args()
```

**Modified Code:**
```python
    parser.add_argument('--single-chunk', help='Validate a single chunk (e.g., chunk_00005)')
    parser.add_argument('--json', action='store_true', help='Output result as JSON')

    args = parser.parse_args()
```

---

### Change 2: JSON Output for Single-Chunk Mode

**File:** `ASR/asr_validator.py`
**Location:** ~Lines 1652-1680 (in the single-chunk validation mode section)

**Original Code:**
```python
    elif args.single_chunk:
        # Single chunk validation mode
        if not args.tts_dir:
            print("Error: --tts-dir required for single-chunk mode")
            sys.exit(1)

        print(f"🎯 Single chunk validation: {args.single_chunk}")

        asr_model, device = load_asr_model_adaptive()
        if not asr_model:
            print("❌ Failed to load ASR model")
            sys.exit(1)

        print(f"✅ ASR model loaded on {device.upper()}")

        result = validate_single_chunk(
            args.single_chunk, Path(args.tts_dir), args.threshold,
            CANON_LOOKUP, asr_model, args.threshold
        )

        score = result.get('score', 0.0)
        passed = result.get('passed', False)

        print(f"\n{'='*50}")
        print(f"Chunk: {args.single_chunk}")
        print(f"Score: {score:.4f}")
        print(f"Status: {'PASSED' if passed else 'FAILED'}")
        print(f"{'='*50}")

        cleanup_asr_model(asr_model)
        sys.exit(0 if passed else 1)
```

**Modified Code:**
```python
    elif args.single_chunk:
        # Single chunk validation mode
        if not args.tts_dir:
            if args.json:
                print('{"error": "--tts-dir required for single-chunk mode"}')
            else:
                print("Error: --tts-dir required for single-chunk mode")
            sys.exit(1)

        if not args.json:
            print(f"🎯 Single chunk validation: {args.single_chunk}")

        asr_model, device = load_asr_model_adaptive()
        if not asr_model:
            if args.json:
                print('{"score": 0.0, "passed": false, "error": "Failed to load ASR model"}')
            else:
                print("❌ Failed to load ASR model")
            sys.exit(1)

        if not args.json:
            print(f"✅ ASR model loaded on {device.upper()}")

        result = validate_single_chunk(
            args.single_chunk, Path(args.tts_dir), args.threshold,
            CANON_LOOKUP, asr_model, args.threshold
        )

        score = result.get('score', 0.0)
        passed = result.get('passed', False)

        if args.json:
            json_output = {
                "chunk_num": args.single_chunk,
                "score": score,
                "passed": passed,
                "ref_text_raw": result.get("ref_text_raw", ""),
                "ref_normalized": result.get("ref_normalized", ""),
                "hyp_text_raw": result.get("hyp_text_raw", ""),
                "hyp_normalized": result.get("hyp_normalized", ""),
                "explanation": result.get("explanation", ""),
                "prose_score": result.get("prose_score", 0.0),
                "id_score": result.get("id_score", 0.0),
                "hallucination_warning": result.get("hallucination_warning", ""),
                "truncation_warning": result.get("truncation_warning", ""),
                "error": result.get("error", "")
            }
            print(json.dumps(json_output))
        else:
            print(f"\n{'='*50}")
            print(f"Chunk: {args.single_chunk}")
            print(f"Score: {score:.4f}")
            print(f"Status: {'PASSED' if passed else 'FAILED'}")
            print(f"{'='*50}")

        cleanup_asr_model(asr_model)
        sys.exit(0 if passed else 1)
```

---

## JSON Output Specification

When `--json` flag is used, the validator outputs a JSON object with the following structure:

```json
{
  "chunk_num": "chunk_XXXXX",
  "score": 0.824,
  "passed": false,
  "ref_text_raw": "STORMCAGE, AD 5147",
  "ref_normalized": "stormcage <ID0>",
  "hyp_text_raw": "Storm Cage, Ed-527.",
  "hyp_normalized": "storm cage ed five hundred and twentyseven",
  "explanation": "substituted: '<ID0>' → 'adi five hundred and twentyseven'",
  "prose_score": 0.456,
  "id_score": 1.0,
  "hallucination_warning": "",
  "truncation_warning": "",
  "error": ""
}
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `chunk_num` | string | Chunk identifier (e.g., "chunk_00001") |
| `score` | float | Combined ASR similarity score (0.0-1.0) |
| `passed` | boolean | Whether chunk meets threshold |
| `ref_text_raw` | string | Original reference text from source |
| `ref_normalized` | string | Normalized reference text |
| `hyp_text_raw` | string | ASR transcribed audio text |
| `hyp_normalized` | string | Normalized transcription |
| `explanation` | string | Explanation of score/differences |
| `prose_score` | float | Prose quality component of score |
| `id_score` | float | ID/entity matching component of score |
| `hallucination_warning` | string | Warning if hallucination detected |
| `truncation_warning` | string | Warning if audio appears truncated |
| `error` | string | Error message if validation failed |

---

## Usage Examples

### Text Output (Backward Compatible)
```bash
ASR/venv/bin/python ASR/asr_validator.py --single-chunk chunk_00006 --tts-dir "Output/Doctor Who/TTS" --threshold 0.85
```
Output:
```
🎯 Single chunk validation: chunk_00006
✅ ASR model loaded on GPU
==================================================
Chunk: chunk_00006
Score: 0.8240
Status: FAILED
==================================================
```

### JSON Output (For Subprocess Integration)
```bash
ASR/venv/bin/python ASR/asr_validator.py --single-chunk chunk_00006 --tts-dir "Output/Doctor Who/TTS" --threshold 0.85 --json
```
Output:
```json
{"chunk_num": "chunk_00006", "score": 0.824, "passed": false, "ref_text_raw": "STORMCAGE, AD 5147", "ref_normalized": "stormcage <ID0>", "hyp_text_raw": "Storm Cage, Ed-527.", "hyp_normalized": "storm cage ed five hundred and twentyseven", "explanation": "substituted: '<ID0>' → 'adi five hundred and twentyseven'", "prose_score": 0.456, "id_score": 1.0, "hallucination_warning": "", "truncation_warning": "", "error": ""}
```

### JSON Output with Audio Override (Regeneration Attempts)
```bash
ASR/venv/bin/python ASR/asr_validator.py \
  --single-chunk chunk_00006 \
  --audio-path "Output/Doctor Who/TTS/audio_chunks/chunk_00006_attempt_1.wav" \
  --tts-dir "Output/Doctor Who/TTS" \
  --threshold 0.85 \
  --json
```

---

## Reapplication Checklist

If `ASR/asr_validator.py` is updated or reset, apply these changes in order:

1. [ ] Locate the argument parser section (around line 1640)
2. [ ] Add `--json` argument after `--single-chunk` argument
3. [ ] Add `--audio-path` argument for single-chunk audio override
3. [ ] Locate the single-chunk validation mode section (around line 1652)
4. [ ] Modify the code to:
    - [ ] Check for `--tts-dir` and output JSON error if missing
    - [ ] Suppress decorative output when `--json` flag is set
    - [ ] Handle ASR model load failure with JSON error
    - [ ] Output JSON result when `--json` flag is set
    - [ ] Preserve original text output when `--json` flag is not set
6. [ ] Test both text and JSON output modes
7. [ ] Verify subprocess integration works in main program

---

## Main Program Integration

The main program (`pocket_tts/audiobook/generator.py`) calls the ASR validator via subprocess:

```python
result = subprocess.run(
    [asr_exe, asr_script,
     '--single-chunk', chunk_num,
     '--tts-dir', str(working_dir),
     '--threshold', str(threshold),
     '--json'],
    capture_output=True,
    text=True,
    timeout=120
)
```

The subprocess output is parsed as JSON and used to determine regeneration quality.

---

## Backward Compatibility

- ✅ Original text output preserved when `--json` flag not used
- ✅ GUI batch validation mode unchanged
- ✅ Monitoring mode unchanged
- ✅ Existing fail.log and asr_failures.json generation unchanged

---

## Last Updated
2026-01-22
