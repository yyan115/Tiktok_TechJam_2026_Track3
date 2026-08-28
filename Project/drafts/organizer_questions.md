# Questions for the Track-3 organizers — SEND TODAY (29 Aug)

Copy-paste ready. Channel: the track's Telegram/Devpost Q&A or
apac-earlycareers@tiktok.com. Answer cutoff for our planning: 30 Aug 12:00
SGT (rental booking); silence => we default to candidate-only evidence for
shapes 6/14 and report multiple MFU conventions side by side.

---

Hi! Questions about Track 3 scoring so we can report results in exactly the
form you want:

1. Per-shape weights: is the final score an equal-weighted sum of the 14
   per-shape MFUs, or weighted (e.g., by FLOPs)?
2. MFU convention: which peak is the denominator (FP32 peak, or the peak of
   the precision actually used internally, e.g., FP16 tensor-core)? Is MFU
   capped at 100%? Our FP32-denominator numbers exceed 100% on some shapes
   when using tensor cores internally.
3. "Bandwidth considered": does that mean memory-bound shapes are credited
   against achievable bandwidth (roofline-style) rather than compute peak?
4. Shapes the official script cannot run on consumer hardware (shape 6 OOMs
   on 8 GB; shape 14's naive attention table is multi-TB on any hardware):
   what evidence is accepted — candidate-only timing + a validated
   alternative reference implementation, or is a baseline comparison
   required on big-memory hardware?
5. Precision: the webinar indicated internal reduced precision is fine if
   outputs meet the stated tolerance — confirming that in writing.
6. Hardware mixing: may different shapes be reported from different
   machines (e.g., most shapes on our own RTX 3060 Ti, shapes 6/14 on a
   rented larger-memory GPU), and if so how should cross-device numbers be
   presented?
7. Implementation constraints: any restriction on kernel language (Triton /
   CUDA C++ / PTX), or on using torch built-ins as fallbacks for edge
   cases, given the no-external-kernel-libraries rule?

Thanks!
