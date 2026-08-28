Created 2026-08-26 4:06PM

The webinar has been held and more information has been received. Here is the transcript of the meeting, along with questions asked at the end. Please piece together the information yourself, where and which questions were asked, if it was answered or not, etc. THIS TRANSCRIPT UNFORTUNATELY MISSED OUT ON THE ENTIRETY OF THE EXPLANATION AND WALKTHROUGH. THEREFORE, IT CONTAINS ONLY TRANSCRIPT FOR SOME PARTS OF THE Q&A. THE SLIDES WALKTHROUGH IS ENTIRELY NOT TRANSCRIBED. EVEN THE Q&A HAS SOME FRAGMENTED ANSWERS UNFORTUNATELY.

In addition, the slides are provided as well, they are ordered correctly.

---

Participants are encouraged to participate in the GitHub,
so we need you to implement and optimize your server
rather than use an already open-sourced project.
Okay, the TensorFlow script defines
the expected default shape sweeps,
but the torch sweep has no equivalent.
Will you share the actual shape combinations
your test against, and do they follow a similar sweep pattern
to the TensorFlow script?
We have changed the problem statement,
and I have provided the test shapes
we will use in the appendix.
You can check it.
What is the business case behind this problem?
Oh, you know, in our daily work,
we are optimizing the structures day by day.
Not only the transformers, but also other structures
in the model, and we also use AI tools to optimize it.
So for this problem statement,
you can try it in your own environment.
The only difference between your work and our work
is just the device you use.
Okay, how would you evaluate this,
since it's not a pure to-Apple country?
Since it's not applied to Apple.
Oh, okay, the final score of the technical execution
I think will be a weighted sum of the MFUs.
So no matter what kind of devices you are using,
the comparison scores are...
I would make it as fair as much as I can.
Also, I will take the bandwidth
into consideration about the execution score.
What is the input scale used in testing,
or it's fixed at once?
Yeah, it's fixed.
I have already provided the test case,
the test shapes into the problem statement.
Okay.
Are you looking more at the GPU optimization process
using AI, or the speed outcome of the optimization?
I think the better result you can outcome
would be better, rather than the speed,
which means you need output higher MFU kernels
rather than code it maybe one hours first.
Okay.
The new 14-shape,
we use the cultural attention mostly for dimension,
but never benchmark default wrong match.
Should we run every appendix or command to be released?
You can run the test
for every appendix row individually, I think.
Yeah.
The default shapes in the script is just a demonstration.
For the 14-shape, what data types should we use?
Flow 32?
What padding ratio or token masking pattern
will be tested?
And for the data type,
the baseline would be used with Flow 32,
and the precision test would also be done
with the Flow 32,
but you can do some quantization during your competition.
Yeah.
Then we only consider about,
we only care about the input and the output precision.
What have you better benchmark
the scoring based on the average speed up,
generally mean performance shape,
performance shape or another formula?
Must every shape has the entry for to receive a score?
Yes.
First of all,
every shape should pass for the precision test
or else it will get a zero point.
And the actual outcome,
the final score would be a combination
of the all shapes,
maybe a weighted MFU, I think.
In terms of all entries evaluated
separately against their baseline or compared to the data.
How are the different time timers and default normalization?
You can only test maybe one of them, I think,
because they are just implemented in different frameworks,
but the actual computation are the same.
So just implement one of them, it would be okay.
Side-end sequence, 100 kilometers.
A dense tension matrix has this shape is many terabytes.
Is data or participants expected to implement
an exam memory efficient chunked attention algorithm?
How will the reference baseline be executed?
Okay.
Yes, the final shape is quite large
for maybe most of the devices.
You need to maybe do some optimizations on that.


---

QUESTIONS ASKED (NOT ALL WERE ANSWERED):
ryan Tan, 06:43
“so will a claude with /loop score the same as an agent with a harness we write?”
Shreyansh Agarwal, 06:44
“You said the test set is already provided, while the written statement says a hidden test will be scored once. Will final judging use the provided dated test rows, a version of those rows with labels hidden, or a completely separate private dataset? Are we prohibited from inspecting the publicly available test labels?”
jeff thomas, 07:12
“I own H200 nodes, can I use it ? is there any preference like we can only use cheap compute (like macbook, rtx, etc)”
NM, 07:12
“Q. The TensorFlow script defines explicit default shape sweeps (batch, qkv, heads, seq_len), but the torch script has no equivalent. Will you share the actual shape combinations you'll test against, and do they follow a similar sweep pattern to the TF script?”
R, 07:12
“what is the business case behind this problem statement?”
jeff thomas, 07:13
“hwo would you evaluation since its not apples to apples comparison on hardware”
Hoang, 07:13
“what is the input scale used in testing or it is fixed at 1”
R, 07:14
“are you looking more at the gpu optimisation process using AI, or the speed outcome of the optimisation?”
Shreyansh Agarwal, 07:14
“The new 14-shape appendix uses causal attention, mostly four layers and FFN dimension equal to QKV dimension, but neither benchmark’s default run matches that. Should we run every appendix row individually, and will an updated runner or exact commands be released?”
Shreyansh Agarwal, 07:15
“For the 14 shapes, what datatype should we use—float32, float16, or bfloat16—and what padding ratio or token-mask patterns will be tested?”
jeff thomas, 07:15
“wouldnt hfu be a better benchmark ?”
Shreyansh Agarwal, 07:16
“Is scoring based on the average speedup, geometric mean, worst-performing shape, or another formula? Must every shape pass for the entry to receive a score?”
Shreyansh Agarwal, 07:17
“Are PyTorch and TensorFlow entries evaluated separately against their own baselines, or compared together? If together, how are their different timers and defaults normalized?”
Vo Khac Trieu, 07:17
“Can you confirm that test shape #14 really has batch size 32 and sequence length 100,000? A dense attention matrix at this shape is many terabytes. Is this a typo, or are participants expected to implement an exact memory-efficient/chunked attention algorithm? How will the reference baseline be executed?”
jeff thomas, 07:18
“can we use different programming languages ?”
Shreyansh Agarwal, 07:18
“Are compilation time and the first run included? Should both baseline and participant implementations use compilation, and what warm-up and repetition settings are official?”
Ed, 07:19
“1. Will judging run the script at its defaults, or with any non-default flags, specifically --compile-baseline? (it is a flag in the provided benchmark script)”
NM, 07:19
“The pytorch script states "The default thresholds are atol=0.001 and rtol=0.01" but the problem statement states "diff should be small enough (relative error < 0.02, abs error < 0.002)." which is correct?”
ac, 07:19
“will we get scored better if we implement using triton/cuda vs pytorch, or will the main objective be weighted sum of mfu while staying within the error margins”
jeff thomas, 07:20
“if its a different language how would it interact with the test script (testing interface).”
jeff thomas, 07:20
“also does the testing system have any memory or compute limits”
Ed, 07:22
“Do judges rerun on their own hardware, and if so on which GPU?”
hb, 07:22
“Are participants permitted to develop, profile, and benchmark their solution using GPUs provided by Singapore’s National Supercomputing Centre (NSCC)? If so, may results obtained on NSCC hardware be included in the technical report, provided we clearly disclose the exact GPU model and environment?”
R, 07:22
“could you go back to the slide on how the kernels would be evaluated (if it is covered)?”
Shreyansh Agarwal, 07:22
“What single calculation converts the 14 results into the performance score? Is correctness on every row mandatory, are speedups capped, and how are failed, timed-out or out-of-memory rows treated?”
jeff thomas, 07:22
“Would MFU be compared directly across different hardware? I’m concerned this could introduce some unfairness between accelerators. For example, a faster GPU may have much higher peak FLOPS, while memory bandwidth or I/O does not scale proportionally. This can make it harder for the faster GPU to achieve a high MFU, whereas a slower GPU may reach a higher MFU more easily even if its actual throughput is lower.”
Hoang, 07:23
“do you rerun the benchmark ?”
ac, 07:23
“are we permitted to use kaggle/ google colab computes for collaboration”
NM, 07:26
“We are supposed to submit results for just a single type of GPU? Our team has diff hardware per person”
hb, 07:26
“For test case 14, is the sequence length of 100,000 correct? With batch size 32 and 16 heads, the full attention matrix would require more than 10 TB of memory even in FP16. How will the reference implementation run this case, and are we expected to use memory-efficient exact attention?”
Sober7135, 07:26
“when will the test case release?”
jeff thomas, 07:27
“any way to stand out ?”