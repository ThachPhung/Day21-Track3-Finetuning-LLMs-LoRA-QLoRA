# Lab 21 — Evaluation Report

**Học viên**: PHUNG_VAN_THACH — 2A202601004  
**Ngày nộp**: 2026-06-25  
**Submission option**: A — Lightweight ZIP

## 1. Setup

- **Base model**: `unsloth/Qwen2.5-3B-bnb-4bit`
- **Method**: QLoRA 4-bit + LoRA adapters, trained with Unsloth and TRL `SFTTrainer`
- **Dataset**: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated`, sampled 200 examples from 52,002 total examples
- **Split**: 180 train examples + 20 eval examples, seed = 42
- **Format**: Alpaca-style text with `instruction_vi`, `input_vi`, `output_vi`
- **Token length**: min = 25, p50 = 227, p95 = 562, p99 = 704, max = 738
- **max_seq_length**: 1024, rounded from p95 and capped for T4 safety
- **GPU**: Tesla T4, 15.6 GB VRAM
- **Training config**: 3 epochs, batch size = 1, gradient accumulation = 8, effective batch size = 8, learning rate = 2e-4, cosine scheduler, `adamw_8bit`, `packing=False`
- **Training cost estimate**: $0.07, approximately 12.4 minutes at $0.35/hour

## 2. Rank Experiment Results

| Rank | Alpha | Trainable Params | Train Time | Peak VRAM | Eval Loss | Perplexity |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 16 | 1,843,200 | 4.07 min | 7.22 GB | 1.557694 | 4.747861 |
| 16 | 32 | 3,686,400 | 4.41 min | 6.62 GB | 1.516083 | 4.554351 |
| 64 | 128 | 14,745,600 | 3.96 min | 8.00 GB | 1.476815 | 4.378976 |
| Base | - | - | - | - | Not measured | Not measured |

The best quantitative result is `r=64`, with the lowest eval loss and perplexity. However, it uses 4x more trainable parameters than `r=16` and 8x more than `r=8`. The improvement from `r=16` to `r=64` is real but modest: perplexity drops from 4.55 to 4.38, around 3.85% relative improvement. The base model was used in qualitative comparison, but base perplexity was not computed in this run.

## 3. Loss Curve Analysis

The saved `loss_curve.png` shows the training loss for the baseline `r=16` adapter. Since eval during training was disabled to avoid T4 OOM, the curve mainly tells us whether optimization was stable, while overfitting must be inferred from final eval loss/perplexity.

For `r=16`, training loss decreased from about 1.61 at step 5 to around 1.39 at step 65, with small fluctuations. This indicates the model was learning without obvious instability. The final eval loss of 1.5161 and perplexity of 4.55 are close enough to the training-loss range that there is no strong evidence of severe overfitting. For `r=64`, the training loss decreased more aggressively, ending around 1.28, and it also achieved the best eval perplexity, so the higher-rank adapter did not obviously overfit on this small eval set. Still, because the eval set only has 20 examples, this conclusion should be treated as directional rather than definitive.

## 4. Qualitative Comparison

### Example 1 — Machine Learning Explanation

**Prompt**: Giải thích khái niệm machine learning cho người mới bắt đầu.

**Base**: The base model explained machine learning as part of AI and described learning from data to make predictions or actions.

**Fine-tuned r=16**: The fine-tuned model also explained machine learning as a computer science field based on improving predictions from data without direct user guidance.

**Judgement**: Improved slightly. The fine-tuned answer is more direct and has a clearer beginner-friendly structure.

### Example 2 — Fibonacci Python Code

**Prompt**: Viết đoạn code Python tính số Fibonacci thứ n.

**Base**: The base model produced a Python function and mentioned recursive or loop-based solutions.

**Fine-tuned r=16**: The fine-tuned model produced a Python function with more explicit input handling, including negative input checks.

**Judgement**: Improved. The fine-tuned answer is more practical because it includes validation and a clearer code-oriented response.

### Example 3 — UI/UX Principles

**Prompt**: Liệt kê 5 nguyên tắc thiết kế UI/UX.

**Base**: The base model gave broad design principles such as user-friendliness, layout, color, fonts, and images.

**Fine-tuned r=16**: The fine-tuned model listed clearer principles such as conversion, adaptation to devices, simplicity, and ease of understanding.

**Judgement**: Improved in format, though some wording is still generic. The fine-tuned output is more list-like and easier to reuse in a report.

### Example 4 — LoRA vs QLoRA

**Prompt**: Tóm tắt sự khác biệt giữa LoRA và QLoRA.

**Base**: The base model correctly recognized LoRA as Low-Rank Adaptation and QLoRA as Quantized LoRA, but the explanation was still vague.

**Fine-tuned r=16**: The fine-tuned answer incorrectly expanded LoRA as "Layer-wise Adaptive Regularization Optimization", which is wrong.

**Judgement**: Degraded. This is an important failure case: fine-tuning on a small generic dataset did not improve factual knowledge and can still produce hallucinations.

### Example 5 — Prompt Engineering, RAG, Fine-tuning

**Prompt**: Phân biệt prompt engineering, RAG, và fine-tuning.

**Base**: The base model described the three methods as different ways to improve model performance.

**Fine-tuned r=16**: The fine-tuned model gave a more instruction-following answer, but still stayed at a high level.

**Judgement**: Slightly improved. The format is cleaner, but the answer should be more precise about RAG handling external knowledge and fine-tuning changing model behavior/style.

## 5. Conclusion về Rank Trade-off

Trong thí nghiệm này, `r=64` cho kết quả perplexity tốt nhất với 4.378976, thấp hơn `r=16` là 4.554351 và `r=8` là 4.747861. Điều này phù hợp với intuition của LoRA: rank cao hơn có nhiều capacity hơn nên mô hình có thể học được nhiều pattern hơn từ dataset. Tuy nhiên, hiệu quả tăng thêm không tuyến tính. Từ `r=8` lên `r=16`, số trainable params tăng 2x và perplexity giảm khoảng 4.1%. Từ `r=16` lên `r=64`, params tăng 4x nhưng perplexity chỉ giảm thêm khoảng 3.85%. Nếu mục tiêu là điểm số eval tốt nhất trong lab, `r=64` là lựa chọn tốt nhất. Nhưng nếu deploy production hoặc cần adapter nhỏ, nhanh, dễ lưu trữ, `r=16` là lựa chọn cân bằng hơn vì chất lượng gần `r=64` nhưng chỉ dùng 3.69M trainable parameters. `r=8` phù hợp khi tài nguyên rất hạn chế, nhưng trong run này chất lượng thấp hơn rõ hơn. Kết luận của tôi là dùng `r=16` cho default production trade-off, và chỉ dùng `r=64` khi cần tối ưu chất lượng thêm và chấp nhận adapter lớn hơn.

Một observation quan trọng là fine-tuning không tự động sửa factual knowledge. Ở prompt LoRA vs QLoRA, model fine-tuned còn hallucinate expansion của LoRA. Điều này củng cố bài học trong Day 21: fine-tuning phù hợp để điều chỉnh style, format, và behavior; còn knowledge gaps nên xử lý bằng RAG hoặc dữ liệu huấn luyện domain-specific chất lượng cao hơn.

## 6. What I Learned

- Rank cao hơn giúp giảm eval loss/perplexity, nhưng lợi ích có diminishing returns so với số trainable parameters tăng thêm.
- Với GPU T4, QLoRA 4-bit + Unsloth giúp train model 3B khá nhanh, tổng thời gian cho 3 adapters chỉ khoảng 12.4 phút.
- Fine-tuning trên dataset nhỏ có thể cải thiện format trả lời nhưng không đảm bảo factual correctness; cần evaluation qualitative để phát hiện hallucination.
