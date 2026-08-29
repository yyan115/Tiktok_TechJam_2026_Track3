RECORDING HAS BEEN UPLOADED AND SO I HAVE TRANSCRIBED THE WEBINAR MEETING. HERE IT IS, SAME FORMAT AS MEETINGNOTES1 WHERE PERSON TALKS FIRST THEN Q&A. Q&A SAME QUESTIONS ITS EXACTLY SAME RECORDING BUT FIXED.

Okay. Hello, everyone. It's time. I'm very glad to introduce the track three problem. We'll invite you to implement a GPU kernel for a transformer layer.
Yeah. Okay. First, let me introduce myself. And you can call me Haoda. And I'm now a machine learning engineer in TikTok video and ML Infra. And I graduated in Tsinghua University and then joined to a startup company and then joined to TikTok in 2020.
And now I'm responsible for the recall system. And in our daily work, we are
going to optimize the model inference and training performance.
Okay. First, let me introduce about the transformer. Maybe most of you have heard about that.
The transformer is a basic component in current machine learning models.
The key point or the key idea of the transformer is the attention, which means a softmax weighted
sum of the value metrics. And currently,
we have seen many fast implementations of transformers on GPUs to optimize the performance
of this structure in the model. For example, the widely used flash attention GitHub has
linked here. But now, I would like to invite you to implement the transformer layer on
your own machine, which means you can first use the AI agent to help you learn the new things.
Faster, for example, what is transformer and how does it compute and how to run it faster.
And then you can implement your own version of the transformer layer and make it runs faster on
your own machines. And also, we have provided some benchmark scripts to you to test your
performance. And here we have some also have some open source projects to build the GPU kernels with
the AI agent, especially for the not only for the transformer layer, but also for other layers,
you can try it. And about how agent works for the kernel generation, that it is totally maybe
regarded as a loop to generate the kernel first, the agent could collect the information over the
current implementation implementation. For example, the basic implementation
from the torch. And then it will diagnose the metrics of the kernel such as how long does it take
to launch the kernel and how much does it take to copy the data and then computing and then it will
implement a better one and then do some benchmark and then profile again. So the I think for most of
the participants, the core challenge is to maybe some of you haven't
get familiar with the CUDA and GPU kernels knowledge, you can learn it with the AI tools
as fast as you can. And then you can improve your implementation with the AI tools.
And, you know, most of the open source projects are built for industry service,
for example, 8100 and H100 or B200. But you know, I think most of you don't have that
chips.
On your own machine. So you just need to implement the kernels for yourself on your
own machine, which means maybe if you use the MacBook, you may have the M3 with the unified
memory, right? So you can you need only to implement the kernels on your own machine and
run it faster on your own machine. And yeah, here we have provided some benchmark script
for you to run for your first demo. You can use all the
open sourced AI tools and and then Android models. And here, we have a simple prompt
for you to implement a demo with the benchmark script and the set-up and install the necessary
dependency of the environment. And you can call the AI agent to install the environment
and then implement the basic version of the of your kernels in the script and run it.
And what you need to submit is that,
first we need you to submit all the source code
you use or you code out or coded by your agent.
And readme file to describe how to maybe compile
or how to run it.
And all the skills that you use during this project
to guide the AI agent to write down the kernels for you.
And also, we need you to submit a tech report
so that you illustrate that what kind of runtime environment
are you using?
And for example, what systems and what kind of devices
and what kind of GPUs are you using?
And how large is the memory or the HPM?
And you also need to describe what kind of AI tools
you have used during this project.
For example, you have used maybe cloud code or codecs
or and what kind of AI tools do you use during this project?
And what kind of AI tools do you have used during this project?
For example, you have used maybe cloud code or codecs
and what kind of models?
For example, maybe a GPT 5.6 or whatever.
And also, we need you to submit some interaction history
between you and your agent.
For example, you are just to submit a sample prompt
to it and then it runs again and again and again.
And then it will provide a good result for you
or you have provided some ideas from you
For example, how large is the memory you have,
and what's the bandwidth of your memory?
And if you have more information to the AI tools,
it may provide much better results to you, I think.
OK, I think the question statement is quite simple.
So just ask me whatever you are interested.
Thanks so much, Hao Dao.
So if you have any questions, feel
free to share in the chat.
H200 nodes can I use?
Is there any preference, like we can only use chip compute?
I think you can use the H200.
And it's industry-level chips.
But what I concern is that for industry-level chips,
there is too much open source implementations in the GitHub.
So we need you to implement.
And optimize your server, rather than use maybe an already open-sourced project.
OK, the TensorFlow script defines the expected default shape
sweeps, but the Torch script has no equivalent.
Will you share the actual shape combinations you test against?
And do they follow a similar sweep pattern to the TLS script?
We haven't changed the problem statement.
And I have provided the test of the shapes we will use in the appendix.
You can check it.
What is the business case behind this problem?
Oh, you know, in our daily work, we are optimizing the structures day by day.
Not only in the transformers, but also other structures in the model.
And we also use AI tools to optimize it.
So for this problem statement, we're going to use the TLS script.
OK.
And you can try it in your own environment.
The only difference between your work and our work is just the device you use.
OK.
How would you evaluate this since it's not appraised to Apple country,
since it is not applied to Apple?
Oh, OK.
The final score over the technical execution,
I think, will be the weighted sum over the MFUs.
So no matter what kind of devices you are using, the cost is the same.
Yes.
Yes.
Yes.
Yes.
Comparison scores are, I would make it as fair as much as I can.
Also, I will take the bandwidth into consideration about the execution score.
What is the input scale used in testing, or it's fixed at once?
Yeah, it's fixed.
I have already provided the test case, the test shapes into the problem statement.
Okay.
Are you looking more at the GPU optimization process using AI, or the speed outcome of the optimization?
I think the better result you can outcome would be better, rather than the speed,
which means you need output higher MFU kernels rather than coding maybe one hours first.
Okay.
A new 14-sheet.
Let me use the cultural attention, mostly for dimensioning, colloquially, but not a Benchmark's default row match.
Should we run every appendix row commands to be released?
You can run the test for every appendix row individually, I think.
Yeah.
The default shapes in the script is just a demonstration for the 14 shapes.
What data types should we use flow to 32, and what padding ratio or token masking pattern will be tested?
And for the data type, the baseline would be used with flow to 32, and the precision test would also be done with the flow to 32.
But you can do some quantization during your computation.
Yeah.
Then, we only...
We only consider about, we only care about the input and the output precision.
Okay.
I feel better.
Benchmark is scoring based on the average speedup.
Generally mean performance shape or another formula.
Must every shape has the entry for to receive a score?
Yes.
First of all, every shape should pass for the precision test.
Or else it will...
get a zero point.
And the actual outcome, the final score would be a combination of the all shapes.
Maybe a weighted MFU, I think.
In terms of entries evaluated separately against their baseline or compared together.
How are the different time and different normalization?
You can only test maybe one of them, I think.
Because the implementation...
They are just implemented.
They are just implemented.
Implemented in different frameworks, but the actual computations are the same.
So just implement one of them would be okay.
Size and sequence of 100 kilometers.
A dense attention matrix as this shape is many terabytes.
Is data or participants expected to implement a memory efficient trunk attention algorithm?
How will the reference baseline be executed?
Okay.
So yes, the final shape is quite large for maybe most of the devices.
So you need to maybe do some optimizations on that.
For example, split the computations maybe into several blocks and then compute it one by one, I think.
And you only need to finish the computation on your own device and submit the text report to us.
Would be okay.
Okay.
Can we use different programming languages?
Yes, you can use Python or whatever you like.
And the only requirement is that the input and output precision check.
Compilation time and the first run included.
No, the compilation time and the first run can be excluded.
Should both baseline and participant implements use compilation?
And what will map and repetition say?
And repetition settings are official.
Usually we'll run the warm-up maybe 20 times.
We're judging run the script at its default or with any non-default flags.
And specifically compile baselines.
Exactly.
You can add whatever flags you like.
The only thing we care about is the precision.
If you can pass the precision check.
Or you can use whatever you like.
Which one?
Oh, you can download the new PyTorch script.
The default threshold has been changed to the same as the problem statement.
Which means relative error less than 0.02 and the absolute error less than 0.002.
Will we get a score better if we implement using code than with PyTorch?
Or will the main objective be weighted sum while staying within the error margins?
Yes.
First of all, all the test cases should be, the precision should be in the required threshold.
And then the score will be counted.
And the final score would be the weighted sum of the MFI.
And you can use whatever technology you can use.
Such as Triton, CUDA, or even maybe lower level languages, whatever you like.
If it's a different language, how would it interact with the test script?
You can write, let me think.
You can write out the output.
You can write down or dump the input and output.
To a file and read the input.
And compare the output with the output of the original Python script.
And if the precision are in the threshold, it will pass the test, I think.
Do judges rerun on their own?
If so, on which GPU?
No, judges will not rerun your script.
Because I think the solution you have written down is specified for your own device.
So that is actually what we want you to do.
Optimize the structure for your own device.
So you need to submit a tech report to us.
Specifies that, what kind of devices are you using?
And how do you reach your best performance?
It's permitted to develop profile on the benchmark with the own solution you contribute.
National.
If so, may any hardware be included in the technical report?
Provided we clearly disclose the exact GPU models.
Ah, Supercomputing Century.
I would suggest you to optimize the performance on your own machine.
Rather than a common provided industrial level chips.
That's what we want.
But if you really want to use that hardware, you can try it.
We will see if you have some innovations in there.
How kernels will be evaluated?
How the kernels will be evaluated?
As I have just described, the final score maybe would be a weighted sum of the MFUs for each test case.
What single correlation converts the multi-resonance into performance score?
Is the correctness on every raw memory a speedup case?
And how are field loads treated?
First of all, you should run and get the right result, right?
Which means you can run it on your own machine and get the right output.
And then, and then we can, you can compare the output with the,
with the baseline.
And if the pre-season can pass, then all things I think is okay.
We compare directly across different hardware.
I'm concerned this could introduce some unfairness between accelerators.
We have much higher peak.
Equal, well done that.
Ah, yes.
So we will not only count it, count it with the MFU.
About the weights, I'm considering.
We will take the bandwidth into account.
And some, if you have, if your hardware has some specific limitations,
you can write down in your, in your tech report.
And we'll take that into account.
Do you re-run benchmark?
No, we will not re-run the benchmark.
Because most of the results are optimized for your own machine, right?
Are we permitted to use Cargo, Google, Columbia Computers?
Yes, you can use that.
But, you know, the benchmark you run with your result, maybe,
I hope that would be your own machine.
Okay.
Is there any other questions here?
Okay.
I'm supposed to submit a result for just a single type of GPU?
Deep hardware.
Yeah.
Just for a single type.
If you have multiple different devices,
you can choose one as the,
as the result that you would optimize for.
14, the sequence is correct.
Which byte size and that's the fold.
Require more than 10 TB of memory even.
How will the reference implementation run?
And are we expected to use memory efficient exact attention?
Yes, you can.
You should design some algorithm to divide the computation
into maybe several blocks and then run it.
But for the baseline part,
we will provide some input and output pairs for you to compare with.
When will the test case release?
For the test,
for the test shapes,
we have already released here.
Yeah.
But for the input and output,
we will release it,
maybe at the final.
Okay.
Is there any other questions?
Standout?
How to put a better kernel as much as we can?
I think.
Okay.
Any last questions for Hao-Da?
Okay.
We'll give it a minute.
Okay.
At 3.29.
Thank you.
Okay.
Thank you.
Okay.
Thanks, Hao-Da.
Thank you so much.
We appreciate your time and your insights.
And thanks everyone for your insightful questions,
you know,
and all the effort that you've been putting into understanding this problem.
