#!/usr/bin/env bash
#
# Sweep the rocFFT kernel config search over a set of lengths, holding
# the VRAM footprint constant so the results are comparable across
# lengths and across GPUs.  Single-precision forward C2C, SBRR.
#
# LENGTH_SET chooses what to sweep:
#
#   mixed     23 mixed-radix lengths (default), described below.
#   powers    the 13 powers of 2, 3 and 5 in [64, 3125] that earlier
#             runs used.  Kept so those runs can be reproduced.
#   both      the union, 34 lengths, as one sweep at one footprint and
#             one set of bounds.  Note that the two sets were not
#             measured with the same elements-per-thread bound, so
#             'both' is not the same thing as 'all' below - it is for
#             when you want the union under bounds you have chosen.
#   holdout   46 lengths that 'mixed' and 'powers' do not contain, for
#             measuring whether a model fitted on those 34 transfers to
#             lengths it has never seen.  Described under "Holdout"
#             below.  Sized to cost about what the existing campaign
#             costs, so it is an overnight run and not a weekend one.
#   long      18 lengths from 3240 to 20480 that none of the above
#             contain, for the one rule the other sets cannot settle.
#             Described under "Long lengths" below.
#   lds       11 lengths already covered by 'powers' and 'mixed',
#             re-swept on purpose: the cheapest set that can separate
#             the two readings of the transforms-per-block rule on a
#             GPU whose LDS budget is not 32 KiB.  Described under
#             "LDS budget" below.  Needs MAX_TPB raised.
#   pp        9 lengths of the form p^2*m for p in 11, 13, 17, which
#             settled how step 3 rounds its target.  Described under
#             "Rounding the ept target" below.  Kept as a regression
#             set now that it has done its job.
#   tpb       15 lengths chosen to place step 7's rounding decision at
#             a spread of asymmetries, so the shape of its penalty can
#             be read off rather than guessed between two forms.
#             Described under "Locating step 7's penalty" below.
#   all       not a single sweep but the whole campaign: the powers set
#             at each of POWER_PCTS, the mixed set at each of
#             MIXED_PCTS, the holdout set at each of HOLDOUT_PCTS and
#             the long set at each of LONG_PCTS, each with the bound
#             its existing numbers were measured with.  This is what
#             brings a new GPU up to parity with the data already
#             collected, in one command.  See "Campaigns" below.
#
# or LENGTHS overrides all of the above with an explicit list.
#
# The mixed set exists because a sweep of pure prime powers cannot
# tell you how to pick a factorization: it only ever offers one
# shape.  Across the 13 powers, the median count of achievable
# elements-per-thread values in the useful 4..16 window is 1, so a
# rule like "take the ept nearest 8" is not being tested, it is being
# handed the answer.  Mixed-radix lengths offer a median of 4, and
# several of the lengths here offer 6.  The same goes for radix
# ordering: permuting 8-8-8-4 moves near-identical radices around,
# whereas 16-9-25 has three genuinely different passes to sequence.
#
# The 23 were picked to cover the choices a model has to make, at the
# least cost in search time, over lengths rocFFT actually ships an
# SBRR kernel for:
#
#   four  single large radix only - 11^2, 13^2, 17^2, and 7*11*13,
#         which has one factorization and three distinct radices.  A
#         regime the power-of-2/3/5 sweep never entered, and almost
#         free to measure.
#   four  odd-only mixes - 105, 675, 1575, 2025.  No radix 2 at all,
#         so none of the power-of-two shortcuts apply.
#   eight two-prime mixes spanning 2*3, 2*5 and 2*7.
#   six   three-prime mixes, which have the densest ept spectra here:
#         1176 and 3000 each offer 12 or 13 distinct ept.
#   two   controls, 729 and 2048, already measured by the earlier
#         sweep, so a re-run isolates the effect of the wider bound
#         below from the effect of changing lengths.
#
# Holdout
# -------
# A model fitted on the 34 lengths above can be checked against those
# same lengths, but that only says it fitted - not that it generalises.
# The holdout set is 46 lengths none of those 34 contain, chosen so the
# answer is informative rather than flattering, and so the sweep costs
# about what the existing campaign costs.
#
# What makes a length informative is that it presents the model with a
# choice.  A length with one factorization, one ordering and two
# achievable threads_per_transform tests almost nothing: whatever rule
# is being evaluated, the answer was handed to it.  Candidate lengths
# were therefore scored by how much freedom each parameter actually has
# at that length, and the set covers every parameter while spending the
# budget where the freedom is.  Five groups, and they are not equal in
# cost - the first two are nearly free:
#
#   5   a large odd radix raised to the 3rd or 4th power - 343, 1331,
#       2197, 2401, 6561.  The existing set has 121, 169 and 289, which
#       are squares and so two passes; these are three and four, with
#       the factorization and its ordering forced.  That isolates the
#       half_lds, direct_to_from_reg and inplace_radix decisions from
#       everything else, and costs about a minute.
#   14  a large odd radix inside a mixture - 143, 187, 325, 363, 507,
#       578, 605, 833, 867, 1156, 1183, 1375, 1734, 2023.  Radix 11, 13
#       and 17 appear in the existing set only as 121, 169, 289 and
#       1001.  These are also where the in-place butterflies matter
#       most: radix 17 drops from 128 VGPRs to 60.
#   9   four distinct primes - 910, 1122, 1155, 1190, 1365, 1430, 1530,
#       1950, 2002.  The existing set reaches three and no further.
#       1155 = 3*5*7*11 has exactly one factorization and 24 orderings
#       of four different radices, which makes it a pure test of how
#       the radices should be sequenced.
#   7   past the end of the existing set - 4000, 4050, 4096, 4704,
#       5488, 6144, 8192, against a previous maximum of 3888.  These
#       are not just bigger: half_lds=0 needs length*tpb*8 bytes of LDS
#       against a 32 KiB budget, so beyond length 4096 it does not fit
#       even at transforms_per_block 1 and the choice disappears.  Five
#       of these seven have no half_lds=0 candidate at all, which is
#       the one regime where a rule fitted on the other 34 cannot
#       apply.
#   11  ordinary highly composite lengths at new sizes - 110, 170, 294,
#       405, 500, 784, 1100, 1300, 1458, 1764, 2916.  These are the
#       closest in character to the existing mixed set and carry most
#       of the budget, because freedom costs search time: they are
#       where the model has the most room to be wrong.
#
# Lengths were taken from the ones rocFFT ships an SBRR kernel for, so
# a result here is a result about something that can actually be
# deployed.  Highly composite small lengths were deliberately passed
# over despite being the richest tests of all: length 480 on its own
# costs more than these 46 together, and 840 two thirds as much, since
# a small length admits every transforms_per_block in every one of its
# many factorizations.
#
# The intended run is the holdout set at all three footprints, which
# the campaign will do on its own:
#
#   LENGTH_SET=all POWER_PCTS= MIXED_PCTS= scripts/tune_sweep.sh
#
# Long lengths
# ------------
# The other three sets stop at 3888, and one of the seven rules a model
# fitted on them lands on cannot be settled from inside that range.
# The rule picks threads_per_transform so that the workgroup comes out
# near 256, which fixes elements per thread at length/256 -- a quantity
# that grows without bound.  Above about 4000 it stops fitting, and the
# fastest kernel measured is consistently the one that buys a wider
# workgroup instead:
#
#   length   fastest measured      what wgs~256 picks
#     4000   tpt 400, ept 10       tpt 250, ept 16
#     4050   tpt 450, ept  9       tpt 405, ept 10
#     4096   tpt 512, ept  8       tpt 256, ept 16
#     4704   tpt 336, ept 14       tpt 294, ept 16
#
# Sweeping the target over 128..1024 said the same thing from both
# sides: above length 3200 a target of 512 beat 256 by 2.4 points on the
# holdout and 3.4 on the training data, and below it 256 won.  But the
# training data had three cells above 3200 out of 180, so the
# improvement did not survive resampling and the constant was not
# identified.  That was not something a better fit could fix; the
# lengths where the answer lives were not in the set.
#
# This set put them in, and the prediction held: on these seven, a
# target of 512 is worth 2.84 points, 95% CI [1.55, 4.32], better in 23
# of 63 cells and worse in one.  The fix it motivated is not a different
# target but a ceiling on ept, since ept is what the rule lets grow
# without bound -- target min(length*tpb/256, 6), which is identical to
# wgs~256 below length 1536 and widens the workgroup only past it.  That
# form is the only one of 33 tried that improves all three sets at once
# (7.232 -> 7.229 training, 9.339 -> 8.776 holdout, 16.232 -> 13.388
# here) and is provably a no-op on every cell below 1536.
#
# The set was meant to have 18 lengths, in two groups: these 7 plus 11
# more from 9216 to 20480, taken from the 471 lengths config_sbrr.py
# carries.  The upper group does not exist.  An SBRR kernel stages the
# whole transform in LDS, so the cheapest it can be is half_lds with
# tpb 1, at 4 bytes an element; against a 32 KiB budget that caps the
# length at exactly 8192, and all 11 came back with zero candidate
# kernels.  They are in the shipped table only as generic entries, which
# the generator handles by deriving tpb itself rather than as a single
# kernel.  The lengths are reachable on gfx950, whose 80 KiB budget puts
# the ceiling at 20480, and the filter below now drops them with a
# message rather than spending a sweep finding out.
#
# Everything here has half_lds free, but above 4096 it would be forced
# to 1, since half_lds=0 needs length*tpb*8 bytes.  The intended run, at
# all three footprints:
#
#   LENGTH_SET=all POWER_PCTS= MIXED_PCTS= HOLDOUT_PCTS= \
#       LONG_PCTS='5 95 200' scripts/tune_sweep.sh
#
# It is not cheap: 13 h per footprint per board, because the 3200-3888
# lengths are highly composite and run 34000 to 60000 candidates each.
#
# Rounding the ept target
# -----------------------
# Four lengths carry most of the holdout regret on the two newest CDNA
# parts - 363, 507, 867 and 1156, all p^2*m for p in 11, 13, 17.  On
# every board the model picks the largest legal threads_per_transform
# and the brute-force optimum picks the middle one:
#
#   867    model tpt=289 (ept 3)     optimum tpt=51 (ept 17)
#   1156   model tpt=289 (ept 4)     optimum tpt=68 (ept 17)
#
# Step 3 then minimised |ept - target| with target min(length*tpb/256, 9),
# and for 867 the reachable epts are 1, 3, 17 and 51, so at target 9 the
# nearest is 3 by six against eight.  The rule misses by less than the
# spacing of the values it has to choose from.  Dense factorizations
# hide this because something always sits near the target; a large prime
# squared exposes it because there is nothing between 3 and 17.  Every
# one of those four optima is the smallest reachable ept not below the
# target, which is a one-word change to step 3: round up, not nearest.
#
# The mechanism was found by looking at the holdout, so the holdout could
# no longer measure it out of sample, and training contains no p^2*m
# length to fit on.  Hence this set, on the same footing as the long set:
# pre-register, sweep lengths nobody has measured, check.
#
# These nine are every unswept p^2*m length on which the two forms
# disagree, which is known exactly rather than guessed because the model
# never reads a timing - the picks on unswept lengths are computable now
# (j85, j94).  The deployed rule wants tpt 121 to 507, round-up wants 22
# to 220, so the two are 2.2x to 6.5x apart in ept on every one of them.
#
#   LENGTH_SET=all POWER_PCTS= MIXED_PCTS= HOLDOUT_PCTS= \
#       PP_PCTS='5 95 200' scripts/tune_sweep.sh
#
# Registered before the sweep: round-up beats the deployed form with a
# 95% CI that excludes zero, and by more on gfx942 and gfx950 (31.4 and
# 39.5 points on the four known lengths) than on gfx90a and gfx1201 (8.2
# and 13.5).  Falsified if the interval straddles zero or the sign flips.
#
# CONFIRMED, 108 cells, run_stamp p2m_new_lengths.  Round-up is 21.05
# points better, 95% CI [15.86, 26.58]; 31.05 down to 10.01; better on
# 88 cells, worse on 19, worst cell 137.0 down to 30.8.  All nine lengths
# picked exactly what the enumeration said they would, sight unseen.  The
# ordering held: 35.5 and 34.1 on gfx950 and gfx942 against 9.6 and 5.0
# on gfx1201 and gfx90a, every interval clear of zero (j96).
#
# Adopted at a cap of 6, where it makes no dataset significantly worse -
# a cap of 9 costs 1.24 points on the long set (j97).  Six of the 34
# shipped entries moved: 243, 675, 1176, 2000, 3000 and 3888.  Outside
# the family the change is a tie, +0.62 over 10 lengths with a CI of
# [-0.74, +1.97], and the gain inside it is 24.6 points over 12 (j98).
# Rounding up costs nothing in registers because slots is
# max(ept, largest radix) and the large prime already sets the array
# width, so it is unchanged on all 552 holdout cells (j95).
#
# Still open, and the reason this section stays.  867 is the one length
# the change did not fix, and tracing it says the next defect is in step
# 7 rather than step 3 (j100).  Step 3's target is a function of tpb, so
# it cannot be settled before step 7, and the narrowing therefore decides
# step 3 separately within each group of candidates sharing a tpb.  On a
# CDNA part the tpb=1 group holds only tpt=289, because tpt=51 would be a
# 51-thread workgroup and a workgroup may not be narrower than a wave -
# 51 is above 32 and below 64, which is exactly why the same length is
# within 1.3% of optimal on gfx1201 and 83% off on gfx942.  Step 7 then
# asks for tpb = 8192/(867*8) = 1.18 and takes 1, selecting the group
# only the wrong tpt can occupy.
#
# Rounding step 7 up the way step 3 now rounds would take tpb=2, which is
# the optimum's own tpb on 867, and it does: tpt=51, tpb=2 on all four
# boards, with the worst cell anywhere falling from 120.5 to 82.2.  It is
# not adopted, because on the data in hand it is a tie on all five
# datasets and pooled - -0.296, 95% CI [-0.747, +0.085] over 996 cells,
# better on gfx942 by 0.63 and worse on gfx90a and gfx950 by about 0.2
# (j101).  One length is not evidence for a rule, and this mechanism was
# again found by reading the holdout.
#
# That sweep is the tpb set below.
#
# Locating step 7's penalty
# -------------------------
# Nearest and round-up are not two rules, they are the ends of one:
#
#   err(tpb) = tpb - t              if tpb >= t
#              lam * (t - tpb)      otherwise
#
# lam=1 is nearest, lam->inf is round-up.  j98 found the step-3 gain
# tracked how far below its target nearest settled rather than the size
# of the step - 3.3 points at 20% below the target against 50.6 at a
# third of it - which is the signature of a finite lam.  A sweep that
# only compares the two ends cannot find it, so this set does something
# else: it places the same decision at different values of lam.
#
# For a single-kernel transform below 1024 the target has a closed form.
# min(8192/(L*el), 1024/tpt) with el=8 is just 1024/L, because tpt <= L
# makes the second term the larger one - checked against the enumeration
# on ten lengths and two boards, exact every time (j102).  So the lam at
# which step 7 flips from floor(t) to ceil(t) is exact too:
#
#   lam* = (ceil(t) - t) / (t - floor(t)),   t = 1024/L
#
# 867 sits at lam* = 4.52.  Sweep a ladder of lengths with lam* spread
# out and the measured optima order it: wherever they stop preferring the
# wider workgroup is lam.  That measures the shape instead of choosing
# between two guesses at it.
#
# The ladder is stratified because lam and the target are confounded - a
# large lam* inside 512 < L < 1024 always means a target near 1, while a
# lam* near 1 can also come from L=77 where the target is 13.3.  So
# stratum A holds the decision at tpb 1 against 2 and varies lam over
# 0.51 to 21.3; stratum B repeats it at tpb 2 against 3 over 0.52 to
# 13.6, which checks the shape is about distance to the target and not
# about tpb=1 in particular.  Length 51 is in for a different reason,
# below.
#
#   LENGTH_SET=all POWER_PCTS= MIXED_PCTS= HOLDOUT_PCTS= \
#       TPB_PCTS='5 95 200' scripts/tune_sweep.sh
#
# Registered before the sweep:
#
#   1. The crossover exists.  Sorted by lam*, the optima prefer ceil(t)
#      at small lam* and floor(t) at large lam*, with at most one sign
#      change.  Falsified if the preference is unordered in lam*, which
#      would mean the penalty is not a function of distance to the target
#      and the whole family is the wrong shape.
#   2. The crossover is above lam* = 4.52, since 867's optimum wants
#      ceil.  Falsified if lengths below 4.52 already prefer floor.
#   3. It is finite, so below 21.3.  If every length prefers ceil then
#      round-up is right as it stands and j101's tie was a power problem
#      rather than a shape problem - which is the outcome that would let
#      step 7 change without introducing a free parameter, and so is the
#      better result of the two.
#
# Deliberately not predicted: the per-board ordering.  867's split needs
# the two boards to see different tpt where step 7 is looking, and of the
# 163 informative unswept lengths exactly one does - length 51, at a
# target of 20 rather than 1.18.  It is in the set as the only available
# probe.  That scarcity is a finding on its own: the 83%-against-1.3%
# split on 867 is close to a one-length accident rather than a general
# wave-size effect, so fixing step 7 should not be expected to pay on
# gfx942 the way fixing step 3 did.
#
# FALSIFIED, 180 cells, run_stamp step_7_sweep.  The ladder came back
# unordered: eight sign changes in the floor-against-ceil preference
# along lam*, and the fraction of cells preferring ceil wanders around
# 0.5 with no relation to lam* at all - 0.92 at lam*=0.84, 0.17 at 4.87,
# 0.83 at 10.1, 0.58 at 18.9.  Both strata agree (0.52 and 0.60), so it
# is not an artefact of one band.  That was the designed falsification
# condition and what it kills is not round-up but the whole family: step
# 7's cost is not a function of the distance from tpb to its target, so
# no reading of that distance is the right shape (j103).
#
# Step 7 therefore stays exactly as deployed.  Pooled over all 1,176
# cells no lam beats lam=1 by more than 0.11 points and nothing clears
# zero except lam=1.25 and lam=1.5, which are a hair from nearest and
# almost certainly noise.
#
# The sweep also corrected the diagnosis that motivated it.  867's regret
# is not step 7's: on all twelve of its cells the optimum is dropped at
# step 4 or step 5, by half_lds=0 or by the ordering rule, before step 7
# chooses anything (j105).  The gfx942 optimum there is half_lds=True,
# which no reading of step 7 can reach.  Rounding step 7 up happened to
# land on the optimum's tpt and tpb while still missing its half_lds,
# which is why it improved the number without addressing the cause - and
# why it did not survive contact with fresh lengths.
#
# What the set did establish, which is more useful than the question it
# was built for: the model keeps the optimum through all seven rules in
# only 55 of 1,176 cells, and in those 55 the regret is exactly zero.  So
# every point of regret the model has is a cell where some rule narrowed
# the optimum away, and the rules are not equally guilty.  By the step
# that drops it first, as a share of all regret:
#
#   3. ept       31.2%     1. dtfr        5.3%
#   4. half_lds  25.9%     7. tpb         4.8%
#   5. ordering  24.4%     6. multiset    1.1%
#   2. inplace    7.3%
#
# Step 7 is the second least important rule in the model by that measure,
# which is also why giving it the right answer recovers so little: 0.64
# points pooled, 7.3% of the 8.71 available (j104).  The three worth
# attacking are ept, half_lds and ordering, together 81%.
#
# That ranking answers which rule should pick better, which is the right
# question only if the model commits to one candidate.  In the intended
# hybrid plan path it does not: it produces a shortlist and rocFFT times
# it.  Scored that way instead - each rule dropped from the chain so the
# benchmark decides it, against the number of kernels that then have to be
# compiled and timed - the order changes completely (j106).
#
# The counts below are per cell, and a cell is one (board, pct, length),
# where pct is the share of last-level cache the batch count is set to
# fill.  So a cell is one concrete problem as a plan sees it - this card,
# this length, this batch size - and a kernel count is the compile and
# time latency one rocfft_plan_create would wait through.  Not length
# plus placement plus direction: neither varies anywhere in this data,
# every measurement is a forward single-precision transform, and the
# inplace rule is the in-place radix butterfly flag rather than the FFT's
# placement.  The tail matters more than the mean, because plan creation
# is latency the caller sees once and not a throughput average.
#
#                                 kernels per plan
#   rule open   values         mean  med  p95  worst    >2  >16  regret  pts/k
#   half_lds    2              1.96    2    2      2     0    0   6.812  1.975
#   multiset    divisors       3.41    2   12     17   481   18   6.853  0.772
#   ept         divisors       5.95    6   12     17  1026    9   5.380  0.674
#   ordering    permutations   7.24    5   24     36   939  100   6.319  0.384
#   inplace     2              1.92    2    2      2     0    0   8.384  0.357
#   dtfr        2              1.96    2    2      2     0    0   8.386  0.339
#   tpb         powers of 2   15.11  9.5   41     64  1011  396   5.158  0.252
#
# '>2' and '>16' are how many of the 1,176 cells need more than that many
# kernels.  For scale, the exhaustive search the model replaces is 6,198
# kernels on the mean cell, 2,544 on the median and 85,368 on the worst.
#
# half_lds is the best buy by a factor of two and a half, and the only
# rule whose cost is bounded: the shortlist is exactly two kernels on
# 1,131 of the 1,176 cells and one on the other 45, where only one value
# is legal.  Never three, at any length, on any board - the rule is
# binary and it is the last thing left undecided, so nothing downstream
# can multiply the count.  That 0.96 of an extra kernel removes 1.900
# points, 95% CI [1.548, 2.296] - 21.8% of the model's regret for one
# more compile.  8.712% becomes 6.812%, the worst cell 120.5 becomes
# 82.2, and 867 leaves the worst list.  So in a hybrid plan the thing to
# do with half_lds is emit both and stop trying to predict it.
#
# The contrast with the rules worth modelling is in the tail rather than
# the mean.  ept averages 5.9 kernels but needs 17 on its worst cell and
# more than two on 1,026 of 1,176; ordering reaches 36; tpb reaches 64
# with 396 cells above sixteen.  Those are the rules whose arity is the
# length's divisor or permutation count, so leaving them open is not a
# shortlist, it is the brute-force search the model exists to replace -
# and a plan whose compile count swings between 2 and 64 depending on how
# the length happens to factor is not a latency budget anyone can hold.
#
# Spending kernels greedily where they buy the most:
#
#   left to the benchmark    mean   med    p95  worst   regret  worst cell
#   nothing                   1.0     1      1      1    8.712       120.5
#   + half_lds                2.0     2      2      2    6.812        82.2
#   + multiset                6.6     4     22     34    5.164        44.9
#   + ept                    34.3    18    106    270    3.664        27.3
#   + tpb                   189.6    88    671   3398    2.071        24.6
#   + dtfr                  269.0   120    932   5097    1.906        15.0
#   + ordering             3503.4  1290  15312  42684    0.057         6.3
#   + inplace              6197.8  2544  24277  85368    0.000         0.0
#
# The knee is the first three rows: two kernels for 6.81%, or about seven
# for 5.16% with a worst case of 34.  Past that the tail runs away far
# faster than the mean - ept moves the average to 34 but the worst cell
# to 270, and tpb to 3,398.  The last row is the exhaustive search
# reaching the zero it has to reach by definition, which is the check
# that the measurement is sound.
#
# For a fixed budget rather than a length-dependent one, the three binary
# rules together are 6.352% at 5.6 kernels on average and never more than
# six, against the eight their arity allows, because the generator only
# supports half_lds together with dtfr and two of the eight combinations
# are illegal.  Most of that is half_lds on its own though: the other two
# binary rules add four kernels to remove a further 0.46 points.
#
# LDS budget
# ----------
# The tuner's LDS allowance is sharedMemPerBlock/2, which is 32 KiB on
# gfx90a, gfx942 and gfx1201 and 80 KiB on gfx950 - a ratio of exactly
# 2.5.  Step 7 of the fitted model is
#
#   tpb = nearest reachable to min(0.25*cap/(length*el), 1024/tpt)
#
# and every board it was fitted on had the same cap, so the data cannot
# tell "a quarter of whatever the budget is" from "a fixed 8 KiB, which
# happens to be a quarter of 32 KiB".  On an 80 KiB board the two read
# 2.5x apart, so one sweep settles it.  It is also the only thing such
# a board answers that the other three have not already answered.
#
# The campaign is the wrong instrument.  gfx950 costs 14 to 28 times
# gfx942 per candidate, which puts the seven sweeps at about 270 hours,
# and worse, it puts single lengths past any plausible walltime: length
# 108 alone is 14 to 27 hours and the sweep checkpoints only between
# lengths, so that length can never complete and the job dies holding
# nothing.
#
# These 11 are the cheap way round, and they are cheap for the same
# reason they are informative.  A prime power or a large prime square
# has one or two factorizations and two to four reachable tpt, so
# almost every candidate measured differs only in tpb - the axis under
# test.  Nine of the 11 read differently under the two hypotheses,
# spanning tpb 3-vs-1 up to 32-vs-13, and 2187 and 3125 are controls
# where both say 1.  The whole set is 1.2 to 2.5 hours per footprint on
# gfx950 with no single length over 45 minutes, so the per-length
# checkpoint actually works.
#
# All 11 are already in 'powers' or 'mixed', which is the point: every
# one has 32 KiB numbers on three boards, so a new board joins a paired
# comparison rather than starting a new baseline.  Note that 121, 169,
# 289 and 1001 have existing data at 95% only.
#
# Candidates the tuner rejects
# ----------------------------
# The tuner checks every candidate against a double-precision reference
# DFT before timing it and drops anything whose relative RMS error is
# over 1e-3, so a sweep that prints INCORRECT lines is reporting a
# codegen defect and not a tuning result.  Across every sweep in this
# file they fall on six lengths and no others: 108, 392 and 784 from the
# earlier sets, and 416, 495 and 616 added by the tpb set - 2,054
# distinct configs there, reproducible on all four boards, so it is not
# one compiler.
#
# The tpb failures are not scattered.  Every one has 11 or 13 as its
# first or second radix and a high elements-per-thread:
#
#   416   13 first or second, tpt  8, ept 52   error 0.294 - 0.429
#   416   13 first or second, tpt 16, ept 26   error 0.0474
#   495   11-5-3-3,           tpt 11, ept 45   error 0.229 - 0.232
#   616   11-7-2-2-2, 11-7-2-4, tpt 11, ept 56 error 0.141 - 0.301
#
# The second line is the useful one: 13-4-2-4 and 13-4-2-2-2 fail at both
# tpt 8 and tpt 16, same factorization, and halving the ept drops the
# error by a factor of six.  So the error is graded in ept rather than
# being a wrong answer from a broken kernel, which points at something
# accumulating or wrapping per element rather than at the butterfly.
#
# Nothing compiles and then silently returns a wrong answer - the tuner
# catches all of it - so this costs coverage rather than correctness: a
# rejected candidate is one the model may well be predicting and the
# sweep cannot score.  Worth filing against the Stockham generator, and
# worth re-checking after any change to it.
#
# MAX_TPB has to be raised or the experiment cannot fail: the default 32
# is below what the scaled reading asks for at length 64 and exactly at
# it for length 81.  64 leaves headroom above the prediction on every
# length here, so the curve can be seen turning over.  The script
# refuses the set otherwise.
#
#   MAX_TPB=64 LENGTH_SET=all POWER_PCTS= MIXED_PCTS= HOLDOUT_PCTS= \
#       LDS_PCTS='5 95 200' scripts/tune_sweep.sh
#
# The follow-up, if the answer is interesting enough to want the tpb
# axis at its widest, is the powers of two - 64 128 256 512 1024 2048,
# six reachable tpt each - at 11 to 22 hours per footprint:
#
#   MAX_TPB=64 LENGTHS='64 128 256 512 1024 2048' scripts/tune_sweep.sh
#
# The batch size is derived from a footprint budget rather than left
# constant, because the tuner allocates an input and an output buffer
# of length*batch single-precision complex elements each - 16 bytes of
# VRAM per element of the transform.  Fixing that total and solving
# for batch keeps the bytes moved the same at every length, so the
# measurement describes the kernel and not the problem size.
#
# FOOTPRINT chooses how the budget is set:
#
#   llc       a share of the GPU's own last level cache, how much set
#   (default) by CACHE_TARGET_PCT - 5% below it by default.  Every
#             part is then measured at the same point relative to its
#             own cache hierarchy, which is what makes a cross-GPU
#             comparison mean anything.
#
#   <MiB>     a fixed size, e.g. FOOTPRINT=15.  Equal bytes moved on
#             every part, but not equal cache pressure: 15 MiB
#             overflows the 8 MiB of L2 an MI250X GCD has while
#             occupying a sixteenth of an MI300X's Infinity Cache, so
#             the same number means "streaming from DRAM" on one part
#             and "resident in cache" on another.
#
# In llc mode the cache size is read from rocminfo rather than
# assumed: it is the largest level the GPU agent reports, which is L3
# on parts with an Infinity Cache / MALL and L2 on parts without one.
# rocminfo reports per agent, so a multi-die board gives the figure
# one HIP device actually sees, and a partitioned GPU gives the
# partition's share.  Parts this was written against, and what they
# should report:
#
#   gfx90a    MI250X             8 MiB   L2 per GCD, no Infinity Cache
#   gfx942    MI300X / MI325X  256 MiB   Infinity Cache
#   gfx950    MI350X / MI355X  256 MiB   Infinity Cache
#   gfx1201   RX 9070 XT        64 MiB   Infinity Cache
#
# Those figures are only cross-checked against what is detected and
# warned about on disagreement; the detected value is what gets used.
#
# The search is bounded two ways, which together cut runtime by orders
# of magnitude.  For length 2048 on gfx950 they found the same winning
# config as an exhaustive search, in 257s instead of over ten hours.
#
# MAX_FACTORS drops long factorizations of small radices, which need
# more passes and have never placed near the top.  It stays at 5 for
# the mixed set, which was checked rather than assumed: allowing a
# 6th radix unlocks no new achievable ept on any of the 23 lengths,
# because the large radices absorb several primes each - 3888 is
# 2^4*3^5, nine prime factors, but 16-27-9 in three.  A 6th radix only
# adds more ways to spell an ept already reachable, and of the 117
# cells measured so far the best 5-radix config averaged 1.9% off the
# winner against 0.9% for 3- and 4-radix ones.  It costs 1.4x to lift.
#
# RADIX_FORM chooses which butterfly implementations to measure.  Radix
# 3, 5, 7, 9, 11, 13 and 17 ship two: the original, which materialises
# its results in temporaries, and an in-place one that rewrites its
# inputs.  The in-place form spends far fewer registers - radix 17 goes
# from 128 VGPRs to 60 on gfx950, radix 11 from 98 to 38, and every
# spill disappears - but each step reads a register the previous step
# just wrote, so it gives the scheduler much less to overlap.  Neither
# wins outright: measured in isolation the in-place form lost by up to
# 23% on an RDNA2 part, yet with workgroup_size and transforms_per_block
# free to adapt the two came within 0.3% of each other, and the in-place
# form won at most values of transforms_per_block.  That is the whole
# reason it is a search dimension and not a decision made in advance.
#
#   both      measure each form (default).  Roughly doubles the space
#             for lengths with an odd radix in the factorization, and
#             leaves pure powers of two untouched.  This is what to use
#             for modelling, because it pairs the two forms at matched
#             wgs/tpt/tpb within a single run, so the comparison carries
#             no cross-run noise.
#   temps     the original form only.  Reproduces the search space as it
#             was before the second form existed, for a clean comparison
#             against data already collected.
#   inplace   the in-place form wherever one exists.
#
# The form that won is reported three ways: an inplace_radix column in
# both CSVs, an _ipRad suffix on the kernel name in the per-candidate
# timing lines, and inplace_radix=True in the winning config line when
# it wins.
#
# MAX_EPT caps elements per thread, which sets a floor on
# threads_per_transform; low-tpt kernels are both the slowest and by
# far the most expensive to compile, since the generator fully unrolls
# the per-thread work.  It is raised from 32 to 64 here.  At 32 the cap
# bound the winner in only 3 of the 117 cells measured, but all three
# were powers of two, where the only ept above 32 is 64 - nothing else
# divides.  Mixed-radix lengths have dense ept spectra, and 17 of the
# 23 have achievable ept between 33 and 64 (36, 40, 42, 45, 48, 49,
# 50, 54, 56, 60), so a cap of 32 would silently remove a region that
# the earlier data gives no grounds to dismiss.  It costs 1.2x to lift.
#
# Expect the mixed set to take roughly twice as long as the 13 powers
# did: 14 min on gfx90a, 65 min on gfx1201 and 2.6 h on gfx950 at
# CACHE_TARGET_PCT=95, scaling with the footprint from there.  Those
# figures predate RADIX_FORM; at the default of 'both' every length with
# an odd radix costs about twice as much again, which is most of the
# mixed set and the powers of 3 and 5, but none of the powers of 2.
#
# The holdout set at all three footprints is predicted to cost 3.1 h on
# gfx90a, 3.8 h on gfx1201 and 8.4 h on gfx942, against 4.1, 4.3 and
# 8.3 h for the existing campaign.  Those predictions come from a cost
# model fitted to the 186 lengths already measured, which reproduces
# each board's campaign total to within 1%; see
# .agent/analysis/j52_holdout_lengths.py.  gfx942 is the board that
# sets the size of the set, because 200% of a 256 MiB cache is 512 MiB
# to move per trial and no amount of search pruning changes that.
#
# Campaigns
# ---------
# LENGTH_SET=all runs several sweeps back to back, because the data
# collected so far is not one sweep but several per GPU, and they do
# not share settings:
#
#   powers   at 5%, 95% and 200% of the LLC, with MAX_EPT=32
#   mixed    at 95% of the LLC, with MAX_EPT=64
#   holdout  at 5%, 95% and 200% of the LLC, with MAX_EPT=64
#
# The bounds differ because the sets were measured at different times
# and the default moved in between; reproducing them is the whole
# point, so the campaign pins each one rather than imposing a single
# value.  MAX_EPT is therefore ignored in this mode - set
# POWER_MAX_EPT, MIXED_MAX_EPT or HOLDOUT_MAX_EPT instead.
#
# Any of the three can be switched off by passing an empty list of
# footprints, which is how you ask for one set on its own: the holdout
# campaign is POWER_PCTS= MIXED_PCTS=, and the old parity campaign,
# from before the holdout set existed, is HOLDOUT_PCTS=.
#
# Each sweep gets its own output directory, and the names are not
# cosmetic: the analysis scripts match the prime-power runs on
# "tuner_sweep_<arch>_llc<pct>_" with no set name in it, and the mixed
# runs on "tuner_sweep_<arch>_mixed_llc<pct>_".  The campaign emits
# both forms so a new GPU's results parse alongside the existing ones
# without renaming anything.
#
#   POWER_PCTS      footprints for the powers set, default "5 95 200"
#   MIXED_PCTS      footprints for the mixed set, default "95"
#   HOLDOUT_PCTS    footprints for the holdout set, default "5 95 200"
#   LONG_PCTS       footprints for the long set, default empty.  The
#                   long set postdates the campaigns already measured,
#                   so it is opt-in: leaving it out keeps a resumed
#                   campaign identical to the one it is resuming.
#   PP_PCTS         footprints for the pp set, default empty, opt-in.
#                   See "Rounding the ept target" above; the whole set
#                   is 13,548 candidates and 25 to 50 minutes per board
#                   at all three, the larger figure being gfx950.
#   TPB_PCTS        footprints for the tpb set, default empty, opt-in.
#                   See "Locating step 7's penalty" above; 63,742
#                   candidates, about 40 minutes per board at all three.
#   LDS_PCTS        footprints for the lds set, default empty, opt-in
#                   for the same reason.  Only worth setting on a GPU
#                   whose LDS allowance is not 32 KiB.
#   POWER_MAX_EPT   bound for the powers set, default 32
#   MIXED_MAX_EPT   bound for the mixed set, default 64
#   HOLDOUT_MAX_EPT bound for the holdout set, default 64
#   LONG_MAX_EPT    bound for the long set, default 64
#   RUN_STAMP       shared suffix for the directory names, default a
#                   timestamp.  Pass the same value again to resume a
#                   campaign: every sweep lands in the directory it
#                   used before and skips the lengths that finished.
#
# Sweeps run in the order listed, so the cheapest footprint goes first
# and a misconfiguration surfaces in minutes rather than hours.  A
# sweep that fails does not stop the campaign; the summary at the end
# reports which ones did.
#
# Usage:
#   scripts/tune_sweep.sh                  # mixed set, 95% of this GPU's LLC
#   LENGTH_SET=all scripts/tune_sweep.sh   # the full seven-sweep campaign
#   LENGTH_SET=all POWER_PCTS= MIXED_PCTS= scripts/tune_sweep.sh
#                                          # just the holdout set, at all
#                                          # three footprints - the model
#                                          # transfer run
#   LENGTH_SET=all DRY_RUN=1 scripts/tune_sweep.sh
#                                          # print all four plans and exit
#   LENGTH_SET=all RUN_STAMP=20261102_1200 scripts/tune_sweep.sh
#                                          # resume that campaign
#   LENGTH_SET=all MIXED_PCTS='5 95 200' scripts/tune_sweep.sh
#                                          # also sweep mixed at the other
#                                          # two footprints, which no GPU
#                                          # has been measured at yet
#   LENGTH_SET=powers MAX_EPT=32 RADIX_FORM=temps scripts/tune_sweep.sh
#                                          # reproduces the earlier runs; MAX_EPT
#                                          # and RADIX_FORM have to be pinned,
#                                          # both defaults moved
#   RADIX_FORM=temps LENGTH_SET=all scripts/tune_sweep.sh
#                                          # the campaign as it was measured
#                                          # before the in-place butterflies
#   LENGTHS='240 1296' scripts/tune_sweep.sh
#   FOOTPRINT=15 scripts/tune_sweep.sh     # fixed 15 MiB instead
#   CACHE_TARGET_PCT=50 scripts/tune_sweep.sh
#   LLC_MIB=256 scripts/tune_sweep.sh      # pretend the LLC is 256 MiB
#   DRY_RUN=1 scripts/tune_sweep.sh        # print the plan and exit
#   TUNER=path/to/rocfft_kernel_config_search scripts/tune_sweep.sh
#
# Pointing OUTDIR at an earlier run resumes it: a length whose log
# already holds a result line is skipped and its numbers are re-read
# from the log, so an interrupted sweep can be continued without
# repeating the lengths that finished.  The default OUTDIR is
# timestamped, so pass it explicitly to resume - or use RUN_STAMP for
# a campaign.  Output lands under OUTROOT, which defaults to .agent
# because that is where the analysis scripts look for it.

set -euo pipefail

TUNER=${TUNER:-build/library/src/rocfft_kernel_config_search}
ROCM_PATH=${ROCM_PATH:-/opt/rocm}

TRIALS=${TRIALS:-20}
FOOTPRINT=${FOOTPRINT:-llc}
CACHE_TARGET_PCT=${CACHE_TARGET_PCT:-95}
LENGTH_SET=${LENGTH_SET:-mixed}
MIN_LENGTH=${MIN_LENGTH:-0}
MAX_LENGTH=${MAX_LENGTH:-0}
MAX_TPB=${MAX_TPB:-32}
MAX_FACTORS=${MAX_FACTORS:-5}
RADIX_FORM=${RADIX_FORM:-both}
# whether the caller pinned MAX_EPT, so campaign mode can say it is ignoring it
max_ept_pinned=${MAX_EPT+yes}
MAX_EPT=${MAX_EPT:-64}
TOP=${TOP:-20}

# campaign settings, used only when LENGTH_SET=all.  Unset substitution
# rather than ":-" so that passing an empty list means "skip this set"
# instead of silently falling back to the default.
POWER_PCTS=${POWER_PCTS-"5 95 200"}
MIXED_PCTS=${MIXED_PCTS-"95"}
HOLDOUT_PCTS=${HOLDOUT_PCTS-"5 95 200"}
# empty by default: the long set was added after the campaigns on the
# existing GPUs ran, and a resume has to reproduce what it resumes
LONG_PCTS=${LONG_PCTS-}
LDS_PCTS=${LDS_PCTS-}
PP_PCTS=${PP_PCTS-}
TPB_PCTS=${TPB_PCTS-}
POWER_MAX_EPT=${POWER_MAX_EPT:-32}
MIXED_MAX_EPT=${MIXED_MAX_EPT:-64}
HOLDOUT_MAX_EPT=${HOLDOUT_MAX_EPT:-64}
LONG_MAX_EPT=${LONG_MAX_EPT:-64}
# 64 to match the holdout set, whose four p^2*m lengths are the paired
# comparison for this one.  At 32 the min_tpt floor removes the very tpt
# the round-up rule is there to reach: 867 would lose tpt=51.
PP_MAX_EPT=${PP_MAX_EPT:-64}
# 64 for the same reason, and it matters more here: the tpb set is designed
# around which tpt step 7 can reach at a given tpb, so a tighter min_tpt floor
# would remove the candidates the whole ladder is built to compare.  Length 51
# needs it most, being the one probe of the wave-size split.
TPB_MAX_EPT=${TPB_MAX_EPT:-64}
# 32 to match the existing numbers for these lengths, which the lds set
# is paired against; its own question is about tpb, not ept
LDS_MAX_EPT=${LDS_MAX_EPT:-32}
# below this the scaled reading of the tpb rule cannot be distinguished
# from a clamp, so the lds set is refused rather than run uninformatively
LDS_MIN_MAX_TPB=64
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}

# where default output directories are created; the analysis scripts
# read from .agent, so that is where a sweep is most useful
OUTROOT=${OUTROOT:-.agent}
# compiling dominates the search and hipRTC serializes it per process,
# so the tuner farms each candidate out to its own process
JOBS=${JOBS:-$(nproc)}
DRY_RUN=${DRY_RUN:-0}

# the tuner's two device buffers, single-precision complex
BYTES_PER_ELEMENT=16

# Pull the gfx name, board name and last level cache out of rocminfo
# for the first GPU agent, as "<arch>|<board>|<llc KiB>".  The CPU
# agent is skipped by device type, so its much larger L3 can't be
# mistaken for the GPU's.
detect_gpu() {
    rocminfo 2>/dev/null | awk '
        function emit() {
            if(!found && typ == "GPU" && llc > 0) {
                print name "|" mkt "|" llc
                found = 1
            }
        }
        /^Agent /            { emit(); name = ""; mkt = ""; typ = ""; llc = 0; next }
        /^  Name:/           { sub(/^  Name:[ \t]*/, ""); sub(/[ \t]+$/, ""); name = $0; next }
        /^  Marketing Name:/ { sub(/^  Marketing Name:[ \t]*/, ""); sub(/[ \t]+$/, ""); mkt = $0; next }
        /^  Device Type:/    { typ = $3; next }
        /^    L[0-9]+:/      { v = $2 + 0; if(v > llc) llc = v; next }
        END                  { emit() }
    '
}

# published figures for the parts above, for cross-checking only
expected_llc_mib() {
    case $1 in
    gfx90a) echo 8 ;;
    gfx942) echo 256 ;;
    gfx950) echo 256 ;;
    gfx1201) echo 64 ;;
    *) echo 0 ;;
    esac
}

# Detected either way: llc mode needs the cache size, and fixed mode
# still wants the arch for the summary header and the output path.
arch=unknown
board=unknown
if command -v rocminfo >/dev/null 2>&1; then
    IFS='|' read -r arch board llc_kib < <(detect_gpu) || true
fi
arch=${arch:-unknown}
board=${board:-unknown}
# drop any target-feature suffix, e.g. gfx90a:sramecc+:xnack-
arch=${arch%%:*}

if [[ -n ${LLC_MIB:-} ]]; then
    llc_kib=$((LLC_MIB * 1024))
    llc_source="LLC_MIB override"
else
    llc_source=rocminfo
fi
llc_mib=$((${llc_kib:-0} / 1024))

if [[ -n ${llc_kib:-} ]] && [[ $llc_source == rocminfo ]]; then
    expected=$(expected_llc_mib "$arch")
    if ((expected > 0 && expected != llc_mib)); then
        printf 'warning: %s usually reports %d MiB of last level cache, detected %d MiB\n' \
            "$arch" "$expected" "$llc_mib" >&2
        printf '         using the detected value; pass LLC_MIB to override\n' >&2
    fi
fi

if [[ $FOOTPRINT == llc ]]; then
    if [[ -z ${llc_kib:-} ]]; then
        echo "error: could not read the last level cache size from rocminfo" >&2
        echo "       pass it explicitly (LLC_MIB=256) or ask for a fixed" >&2
        echo "       footprint instead (FOOTPRINT=15)" >&2
        exit 1
    fi
    if ! [[ $CACHE_TARGET_PCT =~ ^[0-9]+$ ]] || ((CACHE_TARGET_PCT == 0)); then
        echo "error: CACHE_TARGET_PCT must be a positive integer, got '$CACHE_TARGET_PCT'" >&2
        exit 1
    fi
    budget=$((llc_kib * 1024 * CACHE_TARGET_PCT / 100))
    footprint_desc="$CACHE_TARGET_PCT% of $llc_mib MiB last level cache"
    footprint_tag=llc$CACHE_TARGET_PCT
elif [[ $FOOTPRINT =~ ^[0-9]+$ ]] && ((FOOTPRINT > 0)); then
    budget=$((FOOTPRINT * 1024 * 1024))
    footprint_desc="fixed $FOOTPRINT MiB"
    footprint_tag=${FOOTPRINT}mib
else
    echo "error: FOOTPRINT must be 'llc' or a size in MiB, got '$FOOTPRINT'" >&2
    exit 1
fi

case $RADIX_FORM in
both | temps | inplace) ;;
*)
    echo "error: RADIX_FORM must be both, temps or inplace, got '$RADIX_FORM'" >&2
    exit 1
    ;;
esac

if [[ ! -x $TUNER ]]; then
    echo "error: tuner not found or not executable: $TUNER" >&2
    echo "       build it with: ninja -C build rocfft_kernel_config_search" >&2
    echo "       or point TUNER at it" >&2
    exit 1
fi

# A tuner older than this script is the one failure worth pre-empting,
# because of how it presents: the sweep appears to start, then every
# length fails in milliseconds with CLI11 listing the options it did not
# recognise, and the run ends with empty CSVs.  Every option below was
# added to the tuner after the search space grew, so a binary built from
# a branch that predates that rejects all of them while still accepting
# -l and -N.  One --help settles it.
tuner_help=$("$TUNER" brute-force --help 2>&1) || true
tuner_missing=()
for opt in --batchSize --jobs --max-tpb --max-factors --min-tpt --radix-form --top; do
    grep -qF -- "$opt" <<<"$tuner_help" || tuner_missing+=("$opt")
done
if ((${#tuner_missing[@]} > 0)); then
    printf 'error: %s does not understand %s\n' "$TUNER" "${tuner_missing[*]}" >&2
    printf '       It predates those options, so it cannot run this sweep.\n' >&2
    printf '       Rebuild it from a checkout that has them:\n' >&2
    printf '         ninja -C build rocfft_kernel_config_search\n' >&2
    printf '       which also rebuilds rocfft_rtc_helper beside it, and picks\n' >&2
    printf '       up the radix headers the generator embeds - a tuner this old\n' >&2
    printf '       would not know about the in-place butterflies either.\n' >&2
    exit 1
fi

# the tuner spawns rocfft_rtc_helper, which needs the HIP runtime
export LD_LIBRARY_PATH="$ROCM_PATH/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# ------------------------------------------------------------- campaign ----
# LENGTH_SET=all is not a length set but a plan of sweeps, each of which is
# this script re-invoked with a concrete set, footprint and bound.  Running
# them as child processes rather than looping in place means every sweep
# goes through exactly the detection, validation, naming and resume logic
# the single-sweep path uses, with nothing special-cased.
if [[ $LENGTH_SET == all ]]; then
    if [[ -n ${LENGTHS:-} ]]; then
        echo "error: LENGTHS and LENGTH_SET=all are mutually exclusive -- an" >&2
        echo "       explicit length list already says what to sweep" >&2
        exit 1
    fi
    if [[ -n ${OUTDIR:-} ]]; then
        echo "error: OUTDIR cannot be used with LENGTH_SET=all, which writes one" >&2
        echo "       directory per sweep.  Use OUTROOT to choose where they go," >&2
        echo "       or RUN_STAMP to resume an earlier campaign." >&2
        exit 1
    fi
    if [[ $FOOTPRINT != llc ]]; then
        echo "error: LENGTH_SET=all varies the footprint itself, so it needs" >&2
        echo "       FOOTPRINT=llc, got '$FOOTPRINT'" >&2
        exit 1
    fi
    if [[ -n $max_ept_pinned ]]; then
        printf 'warning: MAX_EPT=%s is ignored in campaign mode, where each set\n' \
            "$MAX_EPT" >&2
        printf '         carries its own bound (POWER_MAX_EPT=%s, MIXED_MAX_EPT=%s,\n' \
            "$POWER_MAX_EPT" "$MIXED_MAX_EPT" >&2
        printf '         HOLDOUT_MAX_EPT=%s, LONG_MAX_EPT=%s)\n' \
            "$HOLDOUT_MAX_EPT" "$LONG_MAX_EPT" >&2
    fi

    # Build the plan first so a bad percentage fails now rather than three
    # sweeps in.  The two directory shapes are deliberate: the analysis
    # scripts match prime-power runs without a set name and mixed runs with
    # one, so the campaign has to emit both.
    plan=()
    for pct in $POWER_PCTS; do
        if ! [[ $pct =~ ^[0-9]+$ ]] || ((pct == 0)); then
            echo "error: POWER_PCTS must be positive integers, got '$pct'" >&2
            exit 1
        fi
        plan+=("powers|$pct|$POWER_MAX_EPT|tuner_sweep_${arch}_llc${pct}_${RUN_STAMP}")
    done
    for pct in $MIXED_PCTS; do
        if ! [[ $pct =~ ^[0-9]+$ ]] || ((pct == 0)); then
            echo "error: MIXED_PCTS must be positive integers, got '$pct'" >&2
            exit 1
        fi
        plan+=("mixed|$pct|$MIXED_MAX_EPT|tuner_sweep_${arch}_mixed_llc${pct}_${RUN_STAMP}")
    done
    for pct in $HOLDOUT_PCTS; do
        if ! [[ $pct =~ ^[0-9]+$ ]] || ((pct == 0)); then
            echo "error: HOLDOUT_PCTS must be positive integers, got '$pct'" >&2
            exit 1
        fi
        plan+=("holdout|$pct|$HOLDOUT_MAX_EPT|tuner_sweep_${arch}_holdout_llc${pct}_${RUN_STAMP}")
    done
    for pct in $LONG_PCTS; do
        if ! [[ $pct =~ ^[0-9]+$ ]] || ((pct == 0)); then
            echo "error: LONG_PCTS must be positive integers, got '$pct'" >&2
            exit 1
        fi
        plan+=("long|$pct|$LONG_MAX_EPT|tuner_sweep_${arch}_long_llc${pct}_${RUN_STAMP}")
    done
    for pct in $LDS_PCTS; do
        if ! [[ $pct =~ ^[0-9]+$ ]] || ((pct == 0)); then
            echo "error: LDS_PCTS must be positive integers, got '$pct'" >&2
            exit 1
        fi
        plan+=("lds|$pct|$LDS_MAX_EPT|tuner_sweep_${arch}_lds_llc${pct}_${RUN_STAMP}")
    done
    for pct in $PP_PCTS; do
        if ! [[ $pct =~ ^[0-9]+$ ]] || ((pct == 0)); then
            echo "error: PP_PCTS must be positive integers, got '$pct'" >&2
            exit 1
        fi
        plan+=("pp|$pct|$PP_MAX_EPT|tuner_sweep_${arch}_pp_llc${pct}_${RUN_STAMP}")
    done
    for pct in $TPB_PCTS; do
        if ! [[ $pct =~ ^[0-9]+$ ]] || ((pct == 0)); then
            echo "error: TPB_PCTS must be positive integers, got '$pct'" >&2
            exit 1
        fi
        plan+=("tpb|$pct|$TPB_MAX_EPT|tuner_sweep_${arch}_tpb_llc${pct}_${RUN_STAMP}")
    done
    if ((${#plan[@]} == 0)); then
        echo "error: POWER_PCTS, MIXED_PCTS, HOLDOUT_PCTS, LONG_PCTS," >&2
        echo "       LDS_PCTS, PP_PCTS and TPB_PCTS are all empty," >&2
        echo "       nothing to sweep" >&2
        exit 1
    fi

    printf 'campaign: %d sweeps on %s (%s), stamp %s, radix form %s\n' \
        "${#plan[@]}" "$arch" "$board" "$RUN_STAMP" "$RADIX_FORM"
    printf '%-8s %8s %8s  %s\n' set pct_llc max_ept directory
    for entry in "${plan[@]}"; do
        IFS='|' read -r c_set c_pct c_ept c_dir <<<"$entry"
        printf '%-8s %7s%% %8s  %s\n' "$c_set" "$c_pct" "$c_ept" "$OUTROOT/$c_dir"
    done
    printf 'resume: RUN_STAMP=%s LENGTH_SET=all %s\n\n' "$RUN_STAMP" "$0"

    failed=()
    elapsed_each=()
    campaign_start=$SECONDS
    for entry in "${plan[@]}"; do
        IFS='|' read -r c_set c_pct c_ept c_dir <<<"$entry"
        printf '==== %s set, %s%% of last level cache, max-ept %s\n' \
            "$c_set" "$c_pct" "$c_ept"
        sweep_start=$SECONDS
        if LENGTH_SET=$c_set CACHE_TARGET_PCT=$c_pct MAX_EPT=$c_ept \
            OUTDIR=$OUTROOT/$c_dir "$0"; then
            elapsed_each+=("$((SECONDS - sweep_start))")
        else
            printf 'sweep failed: %s at %s%%, see %s\n' \
                "$c_set" "$c_pct" "$OUTROOT/$c_dir" >&2
            failed+=("$c_set at $c_pct%")
            elapsed_each+=("$((SECONDS - sweep_start))")
        fi
        echo
    done

    if [[ $DRY_RUN != 0 ]]; then
        echo "DRY_RUN set, nothing was run"
        exit 0
    fi

    total=$((SECONDS - campaign_start))
    printf 'campaign finished in %dh%02dm%02ds\n' \
        $((total / 3600)) $(((total % 3600) / 60)) $((total % 60))
    for i in "${!plan[@]}"; do
        IFS='|' read -r c_set c_pct c_ept c_dir <<<"${plan[$i]}"
        secs=${elapsed_each[$i]}
        printf '  %-8s %4s%%  %3dm%02ds  %s\n' \
            "$c_set" "$c_pct" $((secs / 60)) $((secs % 60)) "$OUTROOT/$c_dir"
    done
    if ((${#failed[@]} > 0)); then
        printf '\n%d of %d sweeps failed:\n' "${#failed[@]}" "${#plan[@]}" >&2
        for entry in "${failed[@]}"; do
            printf '  %s\n' "$entry" >&2
        done
        printf 'rerun with RUN_STAMP=%s to retry only what is missing\n' "$RUN_STAMP" >&2
        exit 1
    fi
    exit 0
fi

# Mixed-radix lengths, grouped by prime signature; see the header for
# why these.  All are lengths rocFFT ships an SBRR kernel for.
mixed_lengths=(
    121 169 289 1001          # single large radix: 11^2 13^2 17^2 7*11*13
    105 675 1575 2025         # odd-only: no radix 2 anywhere
    108 972 3888              # 2 and 3
    1000 2000                 # 2 and 5
    392 1568                  # 2 and 7
    150 750 1176 1500 2250 3000   # three primes, densest ept spectra
    729 2048                  # controls, measured by the earlier sweep
)

# Lengths the mixed and powers sets do not contain, for measuring
# whether a model fitted on those transfers; see "Holdout" in the
# header for how these were chosen and what each group is for.  All are
# lengths rocFFT ships an SBRR kernel for, and all have a non-empty
# search space on wave32 and wave64 alike.
holdout_lengths=(
    343 1331 2197 2401 6561        # a large odd radix cubed or 4th power
    143 187 325 363 507 578 605 833 867 1156 1183 1375 1734 2023
                                   # a large odd radix inside a mixture
    910 1122 1155 1190 1365 1430 1530 1950 2002   # four distinct primes
    4000 4050 4096 4704 5488 6144 8192   # past 3888; most lose half_lds=0
    110 170 294 405 500 784 1100 1300 1458 1764 2916
                                   # ordinary highly composite, new sizes
)

# The deployed lengths above 3200 that no other set contains, for the
# workgroup target the other sets cannot identify; see "Long lengths"
# in the header.  All appear in config_sbrr.py.
long_lengths=(
    3240 3375 3456 3600 3645 3750 3840   # the deployed lengths above
                                         # 3200 that no other set has
                                         # and that an SBRR kernel can
                                         # actually reach: 8192 is the
                                         # ceiling at a 32 KiB budget
)

# Lengths whose search space is almost entirely the tpb axis, for
# separating the two readings of the tpb rule on a GPU with a different
# LDS budget; see "LDS budget" in the header.  All are already in the
# powers or mixed set, so each joins a paired comparison.
lds_lengths=(
    81 121 125 169 243 289 625 729 1001   # tpb reads 2.5x apart here
    2187 3125                             # controls: both readings say 1
)

# p^2*m for p in 11, 13, 17, the family that used to carry most of the holdout
# regret on gfx942 and gfx950; see "Rounding the ept target" in the header.
# These nine are every unswept length in the family on which the deployed rule
# and the round-up rule disagreed, so each one was a decision, and all nine
# together are 13,548 candidates because a large prime squared leaves the
# tuner very little to enumerate.  Kept as a regression set: step 3 is the
# rule they settled, so a later change to it has to answer to them.
pp_lengths=(
    242 1452 1815 2420      # 2, 12, 15 and 20 times 11^2
    338 676 2028 2535       # 2, 4, 12 and 15 times 13^2
    1445                    # 5 times 17^2; the only one of its prime that
                            # separates the rules at all, and the cheapest
                            # length in the set at 72 candidates
)

# A ladder that places step 7's floor-against-ceil decision at a spread of
# lam*, the asymmetry at which the choice flips; see "Locating step 7's
# penalty" in the header for what lam* is and why a ladder measures it where
# comparing two rules cannot.  lam* is exact, not estimated, so these lengths
# are a design rather than a sample (j102).  Stratified, because lam* and the
# target itself are confounded across length bands.
tpb_lengths=(
    # A. tpb 1 against 2, target between 1 and 2, lam* 0.51 -> 21.3
    616 663 735 819 875 935 975 980
    # B. tpb 2 against 3, the same shape one rung up, lam* 0.52 -> 13.6
    385 416 455 476 490 495
    # C. the only unswept length on which the two wave sizes leave step 7 a
    #    different set of tpt to choose from, which is 867's mechanism
    51
)

# powers of 2, 3 and 5 in [64, 3125], what the earlier sweeps used
power_lengths=()
for base in 2 3 5; do
    v=$base
    while ((v <= 3125)); do
        ((v >= 64)) && power_lengths+=("$v")
        v=$((v * base))
    done
done

if [[ -n ${LENGTHS:-} ]]; then
    read -r -a lengths <<<"${LENGTHS//,/ }"
    set_desc="explicit"
else
    case $LENGTH_SET in
    mixed) lengths=("${mixed_lengths[@]}") ;;
    powers) lengths=("${power_lengths[@]}") ;;
    holdout) lengths=("${holdout_lengths[@]}") ;;
    long) lengths=("${long_lengths[@]}") ;;
    lds)
        if ((MAX_TPB < LDS_MIN_MAX_TPB)); then
            echo "error: LENGTH_SET=lds needs MAX_TPB at least $LDS_MIN_MAX_TPB," >&2
            echo "       got $MAX_TPB.  The rule it is testing asks for a tpb of" >&2
            echo "       40 at length 64 and 32 at length 81 on an 80 KiB board," >&2
            echo "       so at $MAX_TPB the prediction sits on the bound and the" >&2
            echo "       sweep cannot tell it apart from the clamp." >&2
            exit 1
        fi
        lengths=("${lds_lengths[@]}")
        ;;
    pp) lengths=("${pp_lengths[@]}") ;;
    tpb) lengths=("${tpb_lengths[@]}") ;;
    both) lengths=("${mixed_lengths[@]}" "${power_lengths[@]}") ;;
    *)
        echo "error: LENGTH_SET must be mixed, powers, holdout, long, lds," >&2
        echo "       pp, tpb or both, got '$LENGTH_SET'" >&2
        exit 1
        ;;
    esac
    set_desc=$LENGTH_SET
fi

for length in "${lengths[@]}"; do
    if ! [[ $length =~ ^[0-9]+$ ]] || ((length < 2)); then
        echo "error: not a usable length: '$length'" >&2
        exit 1
    fi
done

mapfile -t lengths < <(printf '%s\n' "${lengths[@]}" | sort -n -u |
    awk -v lo="$MIN_LENGTH" -v hi="$MAX_LENGTH" \
        'NF && (lo == 0 || $1 >= lo) && (hi == 0 || $1 <= hi)')

# An SBRR kernel stages the whole transform in LDS, so it needs
# length * transforms_per_block * bytes_per_element bytes, and the
# cheapest it can possibly be is transforms_per_block 1 with half_lds,
# which is 4 bytes per element.  Past that the tuner enumerates every
# ordering, finds no candidate kernel and reports FAILED, which costs a
# few seconds per length and looks like a bug rather than arithmetic.
# Dropping those lengths up front says so instead.  The budget is
# sharedMemPerBlock/2, which rocminfo does not report, so it is keyed on
# the architecture and can be overridden.
case $arch in
gfx950) LDS_KIB=${LDS_KIB:-80} ;;
*) LDS_KIB=${LDS_KIB:-32} ;;
esac
max_sbrr_length=$((LDS_KIB * 1024 / 4))
mapfile -t dropped < <(printf '%s\n' "${lengths[@]}" |
    awk -v m="$max_sbrr_length" '$1 > m')
if ((${#dropped[@]} > 0)); then
    printf 'skipping %d length(s) with no possible SBRR kernel: %s\n' \
        "${#dropped[@]}" "${dropped[*]}" >&2
    printf '  %d KiB of LDS budget caps a single-kernel transform at %d\n' \
        "$LDS_KIB" "$max_sbrr_length" >&2
    mapfile -t lengths < <(printf '%s\n' "${lengths[@]}" |
        awk -v m="$max_sbrr_length" '$1 <= m')
fi

if ((${#lengths[@]} == 0)); then
    echo "error: no lengths left to sweep after the MIN_LENGTH/MAX_LENGTH" >&2
    echo "       and LDS feasibility filters" >&2
    exit 1
fi

OUTDIR=${OUTDIR:-$OUTROOT/tuner_sweep_${arch}_${set_desc}_${footprint_tag}_$(date +%Y%m%d_%H%M%S)}

batch_for() {
    local length=$1
    local batch=$((budget / (BYTES_PER_ELEMENT * length)))
    ((batch < 1)) && batch=1
    echo "$batch"
}

# the tuner takes a floor on threads_per_transform, but the meaningful
# invariant is elements per thread, which is length/tpt - so derive the
# floor per length rather than fixing it
min_tpt_for() {
    local length=$1
    local tpt=$(((length + MAX_EPT - 1) / MAX_EPT))
    ((tpt < 1)) && tpt=1
    echo "$tpt"
}

# prime factorization as p^e*q^f, so the plan table shows at a glance
# which radix mix each length is exercising
prime_sig() {
    local n=$1 d=2 e out=
    while ((d * d <= n)); do
        e=0
        while ((n % d == 0)); do
            ((++e))
            n=$((n / d))
        done
        ((e == 1)) && out+="${out:+*}$d"
        ((e > 1)) && out+="${out:+*}$d^$e"
        ((++d))
    done
    ((n > 1)) && out+="${out:+*}$n"
    echo "$out"
}

# share of the last level cache a footprint takes, or "-" when the
# cache size isn't known
pct_llc_for() {
    if [[ -z ${llc_kib:-} ]]; then
        echo -
    else
        awk -v b="$1" -v c="$((llc_kib * 1024))" 'BEGIN { printf "%.2f", 100 * b / c }'
    fi
}

printf 'gpu:    %s (%s)' "$arch" "$board"
if [[ -n ${llc_kib:-} ]]; then
    printf ', last level cache %d MiB (%s)' "$llc_mib" "$llc_source"
fi
printf '\n'
printf 'plan:   %s set, %d lengths, %d trials each, %s = %s MiB footprint, -j %d\n' \
    "$set_desc" "${#lengths[@]}" "$TRIALS" "$footprint_desc" \
    "$(awk -v b="$budget" 'BEGIN { printf "%.1f", b / 1048576 }')" "$JOBS"
printf 'bounds: max %d factors, max %d elements per thread\n' "$MAX_FACTORS" "$MAX_EPT"
printf 'radix:  %s butterfly form' "$RADIX_FORM"
case $RADIX_FORM in
both) printf ' -- both measured for radix 3/5/7/9/11/13/17, roughly 2x the space there\n' ;;
temps) printf ' -- reproduces the space from before the in-place forms existed\n' ;;
inplace) printf ' -- in-place only where a second form exists\n' ;;
esac
if [[ -n ${llc_kib:-} ]] && ((budget > llc_kib * 1024)); then
    printf 'note:   the footprint is larger than the last level cache, so this\n'
    printf '        measures the path to DRAM rather than the cache hierarchy\n'
fi
printf '%8s %12s %10s %10s %8s %9s\n' length factors batch MiB pct_llc min_tpt
for length in "${lengths[@]}"; do
    batch=$(batch_for "$length")
    bytes=$((batch * length * BYTES_PER_ELEMENT))
    printf '%8d %12s %10d %10.3f %8s %9d\n' \
        "$length" "$(prime_sig "$length")" "$batch" \
        "$(awk -v b="$bytes" 'BEGIN { print b / 1048576 }')" \
        "$(pct_llc_for "$bytes")" \
        "$(min_tpt_for "$length")"
done

if [[ $DRY_RUN != 0 ]]; then
    echo
    echo "DRY_RUN set, not running"
    exit 0
fi

mkdir -p "$OUTDIR"
summary=$OUTDIR/summary.txt
winners=$OUTDIR/winners.txt
top_csv=$OUTDIR/top_configs.csv
tpb_csv=$OUTDIR/best_per_tpb.csv

# The header below truncates the summary, and a resumed length never
# reaches the line that appends to it, so without this a resume leaves
# the summary describing only the lengths it happened to re-run.  That
# file is the only record of the batch size and candidate count each
# length ran with, and the parser reads both, so the lines are stashed
# and replayed.  winners.txt and the two CSVs need no such care: they
# are rebuilt from every log on every pass.
declare -A prior_summary=()
if [[ -f $summary ]]; then
    while IFS= read -r line; do
        [[ $line =~ ^length[[:space:]]+([0-9]+)[[:space:]] ]] || continue
        prior_summary[${BASH_REMATCH[1]}]=$line
    done <"$summary"
fi

{
    echo "# rocFFT kernel config search sweep"
    echo "# date:      $(date -Is)"
    echo "# tuner:     $TUNER"
    echo "# gpu:       $arch"
    echo "# board:     $board"
    if [[ -n ${llc_kib:-} ]]; then
        echo "# llc:       $llc_mib MiB ($llc_source)"
    fi
    echo "# footprint: $(awk -v b="$budget" 'BEGIN { printf "%.1f", b / 1048576 }') MiB ($footprint_desc)"
    echo "# lengths:   $set_desc (${#lengths[@]}): ${lengths[*]}"
    echo "# trials:    $TRIALS"
    echo "# max-tpb:   $MAX_TPB"
    echo "# max-factors: $MAX_FACTORS"
    echo "# max-ept:   $MAX_EPT"
    echo "# radix-form: $RADIX_FORM"
    echo "# jobs:      $JOBS"
} >"$summary"

echo 'length,rank,time_ms,rel_to_best,workgroup_size,threads_per_transform,transforms_per_block,half_lds,direct_to_from_reg,inplace_radix,factors' >"$top_csv"
echo 'length,transforms_per_block,time_ms,rel_to_best,workgroup_size,threads_per_transform,half_lds,direct_to_from_reg,inplace_radix' >"$tpb_csv"
: >"$winners"

sweep_start=$SECONDS
for length in "${lengths[@]}"; do
    batch=$(batch_for "$length")
    log=$OUTDIR/length_$length.log

    if [[ -s $log ]] && grep -q '^  NS(' "$log"; then
        echo "length $length: already done, skipping"
        if [[ -n ${prior_summary[$length]:-} ]]; then
            printf '%s\n' "${prior_summary[$length]}" >>"$summary"
        else
            # no earlier summary to replay, so reconstruct what can be
            # read back and leave the elapsed time blank rather than
            # inventing one
            printf 'length %-6d batch %-8d %5s  %s\n' \
                "$length" "$batch" "-" \
                "$(grep -oP 'configs_tested,\K[0-9]+' "$log" | tail -1) candidates" \
                >>"$summary"
        fi
    else
        printf 'length %d: batch %d, ' "$length" "$batch"
        start=$SECONDS
        if "$TUNER" brute-force \
            -l "$length" \
            -b "$batch" \
            -N "$TRIALS" \
            -j "$JOBS" \
            --max-tpb "$MAX_TPB" \
            --max-factors "$MAX_FACTORS" \
            --min-tpt "$(min_tpt_for "$length")" \
            --radix-form "$RADIX_FORM" \
            --top "$TOP" \
            >"$log" 2>&1; then
            elapsed=$((SECONDS - start))
            printf '%ds\n' "$elapsed"
            printf 'length %-6d batch %-8d %5ds  %s\n' \
                "$length" "$batch" "$elapsed" \
                "$(grep -oP 'configs_tested,\K[0-9]+' "$log" | tail -1) candidates" >>"$summary"
        else
            elapsed=$((SECONDS - start))
            printf 'FAILED after %ds, see %s\n' "$elapsed" "$log"
            printf 'length %-6d batch %-8d %5ds  FAILED\n' "$length" "$batch" "$elapsed" >>"$summary"
            continue
        fi
    fi

    grep '^  NS(' "$log" >>"$winners" || true

    # The two result tables are already CSV; prepend the length so the
    # whole sweep can be loaded as one frame.  Match on field count to
    # avoid picking up the per-candidate timing lines, which also
    # begin with a number - but take the count from each table's own
    # header rather than pinning it here, because a pinned count
    # silently empties the file the next time the tuner grows a column.
    awk -v L="$length" -F, '
        /^top [0-9]+ of /                   { mode = "top"; next }
        /^best per transforms_per_block/    { mode = "tpb"; next }
        /^rank,/                            { want = NF; next }
        /^transforms_per_block,/            { next }
        mode == "top" && want && NF == want && $1 ~ /^[0-9]+$/ { print L "," $0; next }
        mode == "top"                       { mode = "" }
    ' "$log" >>"$top_csv"

    awk -v L="$length" -F, '
        /^best per transforms_per_block/    { mode = "tpb"; next }
        /^transforms_per_block,/            { want = NF; next }
        mode == "tpb" && want && NF == want && $1 ~ /^[0-9]+$/ { print L "," $0; next }
        mode == "tpb"                       { mode = "" }
    ' "$log" >>"$tpb_csv"
done

# A table whose shape drifted from the header above would land here as an
# empty or ragged file, which is easy to miss and poisons the analysis
# downstream, so check rather than hope.
check_csv() {
    local path=$1 label=$2
    local want rows bad
    want=$(head -1 "$path" | awk -F, '{ print NF }')
    rows=$(($(wc -l <"$path") - 1))
    if ((rows == 0)); then
        printf 'error: %s has a header but no rows -- the tuner output format\n' "$label" >&2
        printf '       no longer matches what this script extracts\n' >&2
        return 1
    fi
    bad=$(awk -F, -v w="$want" 'NR > 1 && NF != w { n++ } END { print n+0 }' "$path")
    if ((bad > 0)); then
        printf 'error: %s: %d of %d rows do not have %d columns\n' \
            "$label" "$bad" "$rows" "$want" >&2
        return 1
    fi
    printf '%-18s %5d rows, %2d columns\n' "$label" "$rows" "$want"
}

csv_ok=0
check_csv "$top_csv" top_configs.csv || csv_ok=1
check_csv "$tpb_csv" best_per_tpb.csv || csv_ok=1

total=$((SECONDS - sweep_start))
printf '\nsweep finished in %dm%02ds\n' $((total / 60)) $((total % 60)) | tee -a "$summary"
echo "results in $OUTDIR/"
echo "  winners.txt       config lines for kernel-generator.py"
echo "  top_configs.csv   the fastest configs per length"
echo "  best_per_tpb.csv  best time at each transforms_per_block"
echo "  length_<N>.log    full output per length"
exit $csv_ok
