"""Generate a six-page English report directly from frozen measured evidence."""
import json
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

ROOT=Path(__file__).resolve().parent

def main():
    r=json.loads((ROOT/'results/summary.json').read_text())
    f=r['frozen'];b=r['benchmark'];c=f['config']
    rows={row['name']:row for row in r['comparisons']}
    selected=rows['final']; initial=rows['baseline']
    diag=json.loads((ROOT/'results/validation_diagnostics.json').read_text())['buckets']
    styles=getSampleStyleSheet()
    for name,size,leading,font,color in [('TitleX',25,29,'Helvetica-Bold','#17334F'),('SubX',11,16,'Helvetica','#567086'),('HeadX',15,19,'Helvetica-Bold','#17334F'),('BodyX',9.8,13.8,'Helvetica','#18232D'),('SmallX',8.0,10.8,'Helvetica','#18232D'),('CodeX',8.0,11.5,'Courier','#18232D')]:
        styles.add(ParagraphStyle(name=name,fontName=font,fontSize=size,leading=leading,textColor=colors.HexColor(color),spaceAfter=10))
    story=[]
    def p(text,style='BodyX'): story.append(Paragraph(text,styles[style]))
    def h(text): p(text,'HeadX')
    def page(): story.append(PageBreak())
    def table(data,widths):
        cells=[[Paragraph(escape(str(x)),styles['SmallX']) for x in row] for row in data]
        t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E8EFF6')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),4),('LINEBELOW',(0,0),(-1,0),0.7,colors.HexColor('#AABCCE')),('LINEBELOW',(0,1),(-1,-1),0.25,colors.HexColor('#DAE2EA'))]))
        story.append(t);story.append(Spacer(1,10))
    improvement=100*(initial['test_bpb']-selected['test_bpb'])/initial['test_bpb']
    p('A Resource-Bounded Hybrid<br/>Small Language Model','TitleX')
    p('DASE7506 MP1 | Student ID: 3036707472 | GitHub: jiucaihehe<br/>Protocol: 7506-mp1-wt2-v2','SubX')
    h('1. Objective and main result')
    p('This project trains small GPT models from random initialization on the supplied WikiText-2 training text. It combines neural prediction, a strictly within-window continuous cache, and compact interpolated Kneser-Ney statistics. Validation selects mixture settings and the final checkpoint. The final predictor and every reported ablation are frozen before any test score is inspected.')
    p(f'The submitted predictor obtains <b>{selected["test_bpb"]:.6f} full-test FP32 BPB</b>, compared with {initial["test_bpb"]:.6f} for the initial classroom baseline ({improvement:.2f}% lower). This total gain includes architecture, training and prediction changes. Matched-weight ablations isolate the contribution of each prediction component; the total improvement is not attributed to one mechanism.')
    table([['Final result','Measured value'],['Validation BPB',f'{selected["validation_bpb"]:.6f}'],['Full-test BPB',f'{selected["test_bpb"]:.6f}'],['CPU scoring / baseline',f'{b["cpu_time_ratio"]:.3f}x (limit 5x)'],['Peak evaluation RAM',f'{b["final_peak_rss_gib"]:.3f} GiB (limit 4 GiB)'],['Uncompressed inference assets',f'{b["inference_assets_mib"]:.3f} MiB (limit 64 MiB)']],[232,255])
    p('The submitted code includes the unchanged course data, tokenizer and scorer, a directly evaluable checkpoint, training and search logs, source/checkpoint hashes, and exact reproduction commands. The method uses no external training text, pretrained weights, cached validation/test answers, cross-window state or evaluation network access.')
    p('The implementation adapts established ideas: continuous caches [1] and Kneser-Ney smoothing [2,3]. The contribution is their constrained implementation, combination, resource accounting and controlled evaluation for this assignment, rather than a claim to invent these methods.')
    page()
    h('2. Predictor and causal boundaries')
    p(f'The selected GPT has width {c["width"]}, depth {c["depth"]}, {c["heads"]} attention heads, learned position embeddings, tied input/output embeddings and vocabulary 2,048. Context length remains 256. Training dropout is {c.get("dropout",0):g}; dropout is disabled during scoring. The neural distribution is softmax(W h[t] / T), with T={c.get("temperature",1):g}.')
    p('<b>Within-window cache.</b> A cache key h[s] is paired with observed token x[s+1]. At query position t, only keys with s &lt; t are eligible, so each cached value is already visible in the input prefix. Similarity weights are a softmax of cosine similarities with optional recency decay:')
    p('a[t,s] = softmax(s &lt; t){theta * cos(h[t],h[s]) - decay*(t-s)}<br/>p_C(w|t) = sum(s &lt; t) a[t,s] * 1{x[s+1] = w}<br/>p_NC = (1-lambda) * p_GPT + lambda * p_C','CodeX')
    p(f'The chosen lambda is {c.get("cache_weight",0):g}, theta is {c.get("cache_theta",16):g}, and decay is {c.get("cache_decay",0):g}. Position zero uses only GPT because the cache is empty. Each window and batch example constructs its own cache on every call. A strict triangular mask excludes future information. There is no persistent evaluation-prefix state.')
    p('<b>Train-derived short-sequence statistics.</b> A 5-gram model is fitted only to the training token sequence. The highest order uses occurrence counts; lower orders use counts of distinct left continuations. The unigram continuation distribution receives a 0.01 pseudocount for finite probabilities. One absolute discount D is used in the recursive interpolated distribution:')
    p('p_KN(w|h) = max(c(h,w)-D,0)/c(h) + b(h)*p_KN(w|suffix(h))','CodeX')
    p('All order-2, order-3 and order-4 events are retained. Order-5 events need raw count at least two. Backoff mass b(h) is one minus the total retained discounted probability, so removed events transfer mass to the lower-order distribution and normalization is preserved. Sorted exact integer keys represent contexts without hash collisions. Keys, pointers, counts and token IDs use bounded integer arrays.')
    p(f'<b>Hybrid mixture.</b> p = (1-alpha) p_NC + alpha p_KN, using D={c.get("ngram_discount",0.75):g} and maximum order {c.get("ngram_order",5)}. The constant alpha is {c.get("ngram_weight",0):g}. When enabled, alpha depends on the longest matched stored context; selected weights for orders 2-5 are {c.get("ngram_weights_by_order",[0,0]+[c.get("ngram_weight",0)]*4)[2:]}. Matching uses only the current prefix. These four finite-grid settings are selected on validation.')
    page()
    h('3. Training, selection and disclosed costs')
    p('All gradient runs use AdamW with weight decay 0.1, 100-step warmup, cosine decay to 10% of peak learning rate and gradient clipping at 1.0. Each update processes 32 sequences of 256 targets. Seed 17 is used throughout. The original baseline and 192-wide experiments use CPU FP32. The regularized 256-wide experiment uses the local Apple M5 GPU through MPS, with validation performed by the unmodified CPU FP32 scorer.')
    p('The 192-wide continuation restarts AdamW at peak learning rate 0.0002 and advances the deterministic sampler past ancestral targets. Other full runs use peak learning rate 0.001. Validation selects saved checkpoints. The 20-step MPS smoke run checks compatibility and is excluded from model selection but included in search cost.')
    training=[['Run','Device','New targets','Cumulative targets','Train seconds']]
    for row in r['training']:
        training.append([row['run'],row['device'],f'{row["new_targets"]:,}',f'{row["cumulative_targets"]:,}',f'{row["train_seconds"]:.1f}'])
    table(training,[151,51,99,104,82])
    p(f'Total new gradient targets: <b>{r["total_new_gradient_targets"]:,}</b>. Logged optimization time: {r["total_train_seconds"]:.1f} s, excluding measured validation scoring. The final checkpoint has {f["train_tokens"]:,} cumulative gradient targets. Reusing checkpoints never resets ancestry. Concurrent CPU/GPU work can affect these elapsed training times; final resource measurements are isolated.')
    p(f'The statistical component scans {r["ngram_build"]["train_targets"]:,} training targets. Its retained build took {r["ngram_build"]["seconds"]:.2f} s. A preliminary, more heavily pruned build took {r["discarded_ngram_build_seconds"]:.2f} s and was replaced using asset-size information before either statistical model was scored. The final stored asset is {r["ngram_build"]["uncompressed_bytes"]/2**20:.3f} MiB uncompressed.')
    p(f'All validation searches together took {r["total_search_seconds"]:.2f} s. Cache search tests 153 deduplicated combinations per backbone: T in {{0.9,1,1.1}}, theta in {{0,4,8,16,32}}, decay in {{0,0.01}}, lambda in {{0,0.05,0.1,0.2,0.3,0.4}}. Statistical mixture search tests 55 combinations of discount {{0.5,0.75,0.9}}, order {{3,4,5}} and weight {{0,0.1,...,0.6}}. Optional gating independently selects one of nine weights {{0,0.1,...,0.8}} in each of four buckets. Search JSON retains every aggregate result; prediction answers are not stored as inference assets.')
    p('The final method is selected solely by full validation BPB among the recorded candidates. Validation has 376,599 scored targets and 1,148,007 bytes; test has 428,405 targets and 1,292,013 bytes. BPB is total natural-log loss divided by ln(2) and the split raw-byte count. Every target except the first is scored exactly once, including the final short window.')
    page()
    h('4. Matched-budget controls and ablations')
    labels={'baseline':'Initial classroom baseline','baseline_cache':'Same initial baseline + cache','uncalibrated_neural':'Selected GPT, T=1','neural_only':'Selected GPT, same T','no_ngram':'Remove statistical mixture','no_cache':'Remove continuous cache','uniform_cache':'Uniform cache, same mixture','constant_ngram':'Constant statistical weight','final':'Final selected hybrid'}
    comparison=[['Frozen predictor','Validation BPB','Test BPB']]
    for row in r['comparisons']:
        comparison.append([labels[row['name']],f'{row["validation_bpb"]:.6f}',f'{row["test_bpb"]:.6f}'])
    table(comparison,[275,106,106])
    p('The initial baseline/cache pair has exactly the same neural weights and 9,830,400 gradient targets. For final-model ablations, all fitted components and their recorded fitting costs are shared: only the designated inference setting changes. Removing the cache sets lambda to zero. Removing the statistical mixture sets alpha to zero. The uniform-cache control sets theta to zero. Constant-mixture control removes the four per-order weights, retaining the previously selected constant alpha. Ablations are not re-tuned on test.')
    p(f'Removing the statistical component changes test BPB from {selected["test_bpb"]:.6f} to {rows["no_ngram"]["test_bpb"]:.6f}; removing the continuous cache changes it to {rows["no_cache"]["test_bpb"]:.6f}. The pure selected GPT scores {rows["neural_only"]["test_bpb"]:.6f}. These matched comparisons identify the value of the components for the submitted checkpoint, without attributing gains to additional gradient training.')
    p('Statistical fitting is disclosed separately from gradient targets because the counting procedure scans the training text. The corresponding inference ablation disables an already-fitted component; it does not erase that shared preprocessing cost. A wider/longer-trained predictor compared with the original baseline is not a clean test of width, dropout or training duration individually.')
    p('Validation diagnostics compare the final model against its no-cache ablation while retaining the same statistical component and interpolation settings. The cache helps when useful continuations recur, but redistributing probability toward cached tokens can hurt new continuations. This is why a small cache weight can outperform a large one.')
    table([['Validation target group','Targets','Nats saved by cache / target']]+[[label,f'{diag[key]["targets"]:,}',f'{diag[key]["nats_saved_per_target"]:+.5f}'] for key,label in [('seen_in_cache','Already in eligible cache'),('not_seen_in_cache','Absent from eligible cache')]], [224,99,164])
    page()
    h('5. Resources, correctness and reproduction')
    table([['Resource','Measured','Limit'],['Median baseline scoring',f'{b["median_cpu_seconds"]["baseline"]:.3f} s','Reference on same host'],['Median final scoring',f'{b["median_cpu_seconds"]["final"]:.3f} s',f'Ratio {b["cpu_time_ratio"]:.3f}x <= 5x'],['Peak whole-process RAM',f'{b["final_peak_rss_gib"]:.3f} GiB','4 GiB'],['Uncompressed inference assets',f'{b["inference_assets_mib"]:.3f} MiB','64 MiB']],[208,122,157])
    p('Timing uses three fresh processes each for baseline and final, alternating on the same Apple M5 CPU with four PyTorch threads, FP32, and no concurrent training. The scorer seconds exclude imports/loading. Peak RAM is whole-process ru_maxrss, including loading, converted using macOS byte units. Compressed NPZ contents are counted at their expanded member sizes. The final checkpoint and all predictor modules are counted; fixed benchmark files, installed libraries and optional experiment evidence are excluded. Results on the instructor\'s Xeon may differ.')
    p('The original contract tests verify causality, normalized probabilities, independent examples, state reset, gradients and exact target coverage. Additional tests compare every prefix with full-window predictions, check strict cache masking, verify normalized pruned statistical distributions, test independent hybrid calls and confirm that dropout is inactive at evaluation. Original fixed files and data are checked against the supplied release hashes.')
    p('Install Python 3.12 and requirements.txt, obtain the exact repository version and matching checkpoint, then run from the repository root:')
    p('python verify_package.py<br/>python -m unittest discover -s tests -v<br/>python evaluate.py --checkpoint artifacts/final.pt &#92;<br/>&nbsp;&nbsp;--device cpu --precision fp32 --threads 4 --split test','CodeX')
    p('No retraining is required. The assets/ngram.npz file is part of the code bundle and required by the final predictor. The README gives full training, cache selection, statistical fitting, gating, freezing and resource-measurement commands. The source and checkpoint hashes recorded in artifacts/FREEZE.json were written before testing. Scores are verified using the same frozen files.')
    p('Final checkpoint SHA-256:<br/>'+f['checkpoint_sha256']['final.pt'],'SmallX')
    p('Freeze timestamp (UTC): '+f['frozen_at_utc'],'SmallX')
    page()
    h('6. Critical analysis, reuse and AI disclosure')
    p('<b>Why combine the components?</b> GPT generalizes across varied contexts. Exact short-sequence statistics provide a complementary estimate for recurring token patterns in the training corpus. A continuous cache adapts within the current document fragment without retaining information between windows. Their probabilities are mixed, preserving finite normalized outputs and the fixed scorer interface.')
    p('<b>Costs and trade-offs.</b> The statistical component uses substantially more inference storage than the small neural checkpoint and adds table-lookup and distribution-combination work. The cache adds an O(B L^2 d) similarity computation, bounded by L=256, with sparse token aggregation. The largest neural candidate increases compute but remains small enough to be evaluated on CPU. The resource audit evaluates all three limits for the same frozen predictor.')
    p('<b>Limits of the evidence.</b> A single random seed is used, so the report does not establish robustness across seeds. Searching validation settings can overfit that split; the test score is a held-out assessment after freezing, not a tuning signal. Width, dropout, training device and schedule were not all independently controlled, so their individual causal effects cannot be separated. Optional per-order gating adds four hyperparameters and its ablation is needed to assess whether the extra selection is worthwhile. Timing and floating-point results may vary on other hardware. Improvements on this corpus and tokenizer do not establish general superiority.')
    p('<b>Reused work.</b> The DASE7506 starter supplies the GPT backbone, training recipe, benchmark, tokenizer, fixed scorer and original tests. model.py, train.py, common.py, evaluate.py, the baseline configuration and benchmark are retained unchanged. Additional trainers adapt the same recipe for checkpoint selection, continuation and MPS. The cache and smoothing concepts are prior work; the constrained implementation and experiment orchestration were written for this submission.')
    p('<b>Substantive AI assistance.</b> OpenAI Codex assisted with planning, implementing the model and experiment scripts, executing local training, testing, selecting settings on validation, analyzing results, drafting the report and reproduction instructions, and preparing submission materials. The student remains responsible for reviewing, understanding and explaining the work. This statement does not assert that independent student review has already occurred.')
    h('References and data attribution')
    p('[1] E. Grave, A. Joulin, N. Usunier. Improving Neural Language Models with a Continuous Cache. ICLR 2017. https://arxiv.org/abs/1612.04426<br/>[2] R. Kneser, H. Ney. Improved Backing-Off for M-Gram Language Modeling. ICASSP 1995, pp. 181-184. https://www-i6.informatik.rwth-aachen.de/publications/download/951/Kneser-ICASSP-1995.pdf<br/>[3] S. F. Chen, J. Goodman. An Empirical Study of Smoothing Techniques for Language Modeling. Computer Speech and Language 13(4), 1999, pp. 359-394.<br/>[4] S. Merity, C. Xiong, J. Bradbury, R. Socher. Pointer Sentinel Mixture Models. ICLR 2017. https://arxiv.org/abs/1609.07843<br/>[5] DASE7506 MP1 supplied GUIDE.md, README.md and protocol 7506-mp1-wt2-v2.','SmallX')
    p('WikiText text is by Wikipedia contributors. Retain the supplied CC BY-SA 3.0 and GNU Free Documentation License notices and dataset revision attribution when redistributing it. These data notices do not relicense the surrounding classroom source.','SmallX')
    out=ROOT/'output/pdf';out.mkdir(parents=True,exist_ok=True)
    def footer(canvas,doc):
        canvas.setStrokeColor(colors.HexColor('#D6E0EA'));canvas.line(54,43,541,43)
        canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#567086'))
        canvas.drawString(54,30,'DASE7506 MP1 | 3036707472 | Frozen CPU FP32 evaluation')
        canvas.drawRightString(541,30,str(doc.page))
    doc=SimpleDocTemplate(str(out/'report.pdf'),pagesize=(595.28,841.89),leftMargin=54,rightMargin=54,topMargin=46,bottomMargin=57,title='A Resource-Bounded Hybrid Small Language Model',author='3036707472')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    (ROOT/'report.pdf').write_bytes((out/'report.pdf').read_bytes())
    print(out/'report.pdf')

if __name__=='__main__':main()
