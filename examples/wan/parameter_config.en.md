# Parameter Config

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-05T08:17:06.211Z pushedAt=2026-06-09T02:35:26.293Z -->

This document introduces the weights and parameter configuration of the Wan2.1 model.

## Model Weights

Detailed information on model weights is shown in the table. Users need to set the weight path themselves (e.g., `/home/_{username}_/Wan2.1-T2V-14B`).

**Table 1**  Model weight list

<a name="table822517510017"></a>
<table><thead align="left"><tr id="row42261751705"><th class="cellrowborder" valign="top" width="16.11%" id="mcps1.2.4.1.1"><p id="p13172172254"><a name="p13172172254"></a><a name="p13172172254"></a>Model</p>
</th>
<th class="cellrowborder" valign="top" width="34.02%" id="mcps1.2.4.1.2"><p id="p17172322511"><a name="p17172322511"></a><a name="p17172322511"></a>Description</p>
</th>
<th class="cellrowborder" valign="top" width="49.87%" id="mcps1.2.4.1.3"><p id="p15172102851"><a name="p15172102851"></a><a name="p15172102851"></a>Weights</p>
</th>
</tr>
</thead>
<tbody><tr id="row11263114101711"><td class="cellrowborder" valign="top" width="16.11%" headers="mcps1.2.4.1.1 "><p id="p526304101710"><a name="p526304101710"></a><a name="p526304101710"></a>Wan2.1-T2V-14B</p>
</td>
<td class="cellrowborder" valign="top" width="34.02%" headers="mcps1.2.4.1.2 "><p id="p14263174141711"><a name="p14263174141711"></a><a name="p14263174141711"></a>Text-to-video model</p>
</td>
<td class="cellrowborder" valign="top" width="49.87%" headers="mcps1.2.4.1.3 "><p id="p2026319415173"><a name="p2026319415173"></a><a name="p2026319415173"></a>Click the <a href="https://huggingface.co/Wan-AI/Wan2.1-T2V-14B/tree/main" target="_blank" rel="noopener noreferrer">link</a> to get the weight file.</p>
</td>
</tr>
<tr id="row181291045151718"><td class="cellrowborder" valign="top" width="16.11%" headers="mcps1.2.4.1.1 "><p id="p8129145141713"><a name="p8129145141713"></a><a name="p8129145141713"></a>Wan2.1-I2V-14B-480P</p>
</td>
<td class="cellrowborder" valign="top" width="34.02%" headers="mcps1.2.4.1.2 "><p id="p101291445171712"><a name="p101291445171712"></a><a name="p101291445171712"></a>Image-to-video model</p>
</td>
<td class="cellrowborder" valign="top" width="49.87%" headers="mcps1.2.4.1.3 "><p id="p6129144531718"><a name="p6129144531718"></a><a name="p6129144531718"></a>Click the <a href="https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P/tree/main" target="_blank" rel="noopener noreferrer">link</a> to get the weight file.</p>
</td>
</tr>
<tr id="row1623154911176"><td class="cellrowborder" valign="top" width="16.11%" headers="mcps1.2.4.1.1 "><p id="p4232104911715"><a name="p4232104911715"></a><a name="p4232104911715"></a>Wan2.1-I2V-14B-720P</p>
</td>
<td class="cellrowborder" valign="top" width="34.02%" headers="mcps1.2.4.1.2 "><p id="p1232204951711"><a name="p1232204951711"></a><a name="p1232204951711"></a>Image-to-video model</p>
</td>
<td class="cellrowborder" valign="top" width="49.87%" headers="mcps1.2.4.1.3 "><p id="p11232154961717"><a name="p11232154961717"></a><a name="p11232154961717"></a>Click the <a href="https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P/tree/main" target="_blank" rel="noopener noreferrer">link</a> to get the weight file.</p>
</td>
</tr>
</tbody>
</table>

## Model Parameters

Users can set the model parameters in the inference script. For details on parameter explanations, see the table.

**Table 2**  Model inference parameter description

<a name="table8470029931"></a>
<table><thead align="left"><tr id="row347116291633"><th class="cellrowborder" valign="top" width="21.060000000000002%" id="mcps1.2.4.1.1"><p id="p184601755194118"><a name="p184601755194118"></a><a name="p184601755194118"></a>Parameter</p>
</th>
<th class="cellrowborder" valign="top" width="18.93%" id="mcps1.2.4.1.2"><p id="p7460155516416"><a name="p7460155516416"></a><a name="p7460155516416"></a>Description</p>
</th>
<th class="cellrowborder" valign="top" width="60.01%" id="mcps1.2.4.1.3"><p id="p84608550417"><a name="p84608550417"></a><a name="p84608550417"></a>Value</p>
</th>
</tr>
</thead>
<tbody><tr id="row1147114291237"><td class="cellrowborder" valign="top" width="21.060000000000002%" headers="mcps1.2.4.1.1 "><p id="p2037213644411"><a name="p2037213644411"></a><a name="p2037213644411"></a>model_base</p>
</td>
<td class="cellrowborder" valign="top" width="18.93%" headers="mcps1.2.4.1.2 "><p id="p1637233617442"><a name="p1637233617442"></a><a name="p1637233617442"></a>Weight path</p>
</td>
<td class="cellrowborder" valign="top" width="60.01%" headers="mcps1.2.4.1.3 "><p id="p11372153624420"><a name="p11372153624420"></a><a name="p11372153624420"></a>Path  to the model weights.</p>
</td>
</tr>
<tr id="row1392552918328"><td class="cellrowborder" valign="top" width="21.060000000000002%" headers="mcps1.2.4.1.1 "><p id="p12925172953215"><a name="p12925172953215"></a><a name="p12925172953215"></a>task</p>
</td>
<td class="cellrowborder" valign="top" width="18.93%" headers="mcps1.2.4.1.2 "><p id="p12925182933218"><a name="p12925182933218"></a><a name="p12925182933218"></a>Task type</p>
</td>
<td class="cellrowborder" valign="top" width="60.01%" headers="mcps1.2.4.1.3 "><p id="p1292502910324"><a name="p1292502910324"></a><a name="p1292502910324"></a><code>t2v-14B</code> or <code>i2v-14B</code>.</p>
</td>
</tr>
<tr id="row12468867107"><td class="cellrowborder" valign="top" width="21.060000000000002%" headers="mcps1.2.4.1.1 "><p id="p194681468109"><a name="p194681468109"></a><a name="p194681468109"></a>size</p>
</td>
<td class="cellrowborder" valign="top" width="18.93%" headers="mcps1.2.4.1.2 "><p id="p64681068102"><a name="p64681068102"></a><a name="p64681068102"></a>Video resolution</p>
</td>
<td class="cellrowborder" valign="top" width="60.01%" headers="mcps1.2.4.1.3 "><p id="p20345194662814"><a name="p20345194662814"></a><a name="p20345194662814"></a>Width*height of the generated video.</p>
<a name="ul172121649202811"></a><a name="ul172121649202811"></a><ul id="ul172121649202811"><li><code>t2v-14B</code>: The default value is <code>1280*720</code>.</li><li><code>i2v-14B-480P</code>: The default value is <code>[832, 480]</code> and <code>[720, 480]</code>.</li><li><code>i2v-14B-720P</code>: The default value is <code>[1280, 720]</code>
.</li></ul>
</td>
</tr>
<tr id="row4174145417181"><td class="cellrowborder" valign="top" width="21.060000000000002%" headers="mcps1.2.4.1.1 "><p id="p8174195491814"><a name="p8174195491814"></a><a name="p8174195491814"></a>frame_num</p>
</td>
<td class="cellrowborder" valign="top" width="18.93%" headers="mcps1.2.4.1.2 "><p id="p41741154181816"><a name="p41741154181816"></a><a name="p41741154181816"></a>Number of frames in generated video</p>
</td>
<td class="cellrowborder" valign="top" width="60.01%" headers="mcps1.2.4.1.3 "><p id="p17174185410180"><a name="p17174185410180"></a><a name="p17174185410180"></a>The default value is <code>81</code> frames.</p>
</td>
</tr>
<tr id="row180313214350"><td class="cellrowborder" valign="top" width="21.060000000000002%" headers="mcps1.2.4.1.1 "><p id="p6804721153516"><a name="p6804721153516"></a><a name="p6804721153516"></a>sample_steps</p>
</td>
<td class="cellrowborder" valign="top" width="18.93%" headers="mcps1.2.4.1.2 "><p id="p158042021163512"><a name="p158042021163512"></a><a name="p158042021163512"></a>Sampling steps</p>
</td>
<td class="cellrowborder" valign="top" width="60.01%" headers="mcps1.2.4.1.3 "><p id="p178041921173514"><a name="p178041921173514"></a><a name="p178041921173514"></a>Number of iterative denoising steps for the diffusion model. The default value is <code>50</code> for the t2v model and <code>40</code> for the i2v model.</p>
</td>
</tr>
<tr id="row1235851163710"><td class="cellrowborder" valign="top" width="21.060000000000002%" headers="mcps1.2.4.1.1 "><p id="p1535801143715"><a name="p1535801143715"></a><a name="p1535801143715"></a>prompt</p>
</td>
<td class="cellrowborder" valign="top" width="18.93%" headers="mcps1.2.4.1.2 "><p id="p11358214377"><a name="p11358214377"></a><a name="p11358214377"></a>Text prompt</p>
</td>
<td class="cellrowborder" valign="top" width="60.01%" headers="mcps1.2.4.1.3 "><p id="p12358181183714"><a name="p12358181183714"></a><a name="p12358181183714"></a>User-defined, used to control video generation.</p>
</td>
</tr>
<tr id="row1476210452117"><td class="cellrowborder" valign="top" width="21.060000000000002%" headers="mcps1.2.4.1.1 "><p id="p57621342211"><a name="p57621342211"></a><a name="p57621342211"></a>image</p>
</td>
<td class="cellrowborder" valign="top" width="18.93%" headers="mcps1.2.4.1.2 "><p id="p147625412111"><a name="p147625412111"></a><a name="p147625412111"></a>Image path for video generation</p>
</td>
<td class="cellrowborder" valign="top" width="60.01%" headers="mcps1.2.4.1.3 "><p id="p11762748216"><a name="p11762748216"></a><a name="p11762748216"></a>Required for i2v model inference. User-defined, used to control video generation.</p>
</td>
</tr>
<tr id="row1046211199392"><td class="cellrowborder" valign="top" width="21.060000000000002%" headers="mcps1.2.4.1.1 "><p id="p5462151973911"><a name="p5462151973911"></a><a name="p5462151973911"></a>base_seed</p>
</td>
<td class="cellrowborder" valign="top" width="18.93%" headers="mcps1.2.4.1.2 "><p id="p54621819193910"><a name="p54621819193910"></a><a name="p54621819193910"></a>Random seed</p>
</td>
<td class="cellrowborder" valign="top" width="60.01%" headers="mcps1.2.4.1.3 "><p id="p15462161912392"><a name="p15462161912392"></a><a name="p15462161912392"></a>Random seed used for video generation.</p>
</td>
</tr>
<tr id="row1321483517395"><td class="cellrowborder" valign="top" width="21.060000000000002%" headers="mcps1.2.4.1.1 "><p id="p22151835183910"><a name="p22151835183910"></a><a name="p22151835183910"></a>use_attentioncache</p>
</td>
<td class="cellrowborder" valign="top" width="18.93%" headers="mcps1.2.4.1.2 "><p id="p1421543511397"><a name="p1421543511397"></a><a name="p1421543511397"></a>Whether to enable attentioncache algorithm optimization</p>
</td>
<td class="cellrowborder" valign="top" width="60.01%" headers="mcps1.2.4.1.3 "><p id="p485895083013"><a name="p485895083013"></a><a name="p485895083013"></a>This optimization is lossy. If enabled, the following parameters must be set: <code>start_step</code>, <code>attentioncache_interval</code>, and <code>end_step</code>.</p>
<a name="ul12436145316300"></a><a name="ul12436145316300"></a><ul id="ul12436145316300"><li><code>start_step</code>: The step at which the cache starts.</li><li><code>attentioncache_interval</code>: Number of consecutive caches.</li><li><code>end_step</code>: The step at which the cache ends.</li></ul>
</td>
</tr>
<tr id="row185991037277"><td class="cellrowborder" valign="top" width="21.060000000000002%" headers="mcps1.2.4.1.1 "><p id="p76004312711"><a name="p76004312711"></a><a name="p76004312711"></a>nproc_per_node</p>
</td>
<td class="cellrowborder" valign="top" width="18.93%" headers="mcps1.2.4.1.2 "><p id="p1460011372711"><a name="p1460011372711"></a><a name="p1460011372711"></a>Number of parallel cards</p>
</td>
<td class="cellrowborder" valign="top" width="60.01%" headers="mcps1.2.4.1.3 "><a name="ul6979743282"></a><a name="ul6979743282"></a><ul id="ul6979743282"><li>Wan2.1-T2V-14B supports 1, 2, 4, or 8 cards.</li><li>Wan2.1-I2V-14B supports 1, 2, 4, or 8 cards.</li></ul>
</td>
</tr>
<tr id="row16261195693912"><td class="cellrowborder" valign="top" width="21.060000000000002%" headers="mcps1.2.4.1.1 "><p id="p2261155643910"><a name="p2261155643910"></a><a name="p2261155643910"></a>ulysses_size</p>
</td>
<td class="cellrowborder" valign="top" width="18.93%" headers="mcps1.2.4.1.2 "><p id="p13261256153911"><a name="p13261256153911"></a><a name="p13261256153911"></a>Ulysses parallel size</p>
</td>
<td class="cellrowborder" valign="top" width="60.01%" headers="mcps1.2.4.1.3 "><p id="p1526135612397"><a name="p1526135612397"></a><a name="p1526135612397"></a>The default value is <code>1</code>. <code>ulysses_size * cfg_size = nproc_per_node</code>.</p>
</td>
</tr>
<tr id="row111392315243"><td class="cellrowborder" valign="top" width="21.060000000000002%" headers="mcps1.2.4.1.1 "><p id="p1711482312419"><a name="p1711482312419"></a><a name="p1711482312419"></a>cfg_size</p>
</td>
<td class="cellrowborder" valign="top" width="18.93%" headers="mcps1.2.4.1.2 "><p id="p13114162312249"><a name="p13114162312249"></a><a name="p13114162312249"></a>CFG parallel size</p>
</td>
<td class="cellrowborder" valign="top" width="60.01%" headers="mcps1.2.4.1.3 "><p id="p01141523162419"><a name="p01141523162419"></a><a name="p01141523162419"></a>The default value is <code>1</code>. <code>ulysses_size * cfg_size = nproc_per_node</code>.</p>
</td>
</tr>
<tr id="row1259012559561"><td class="cellrowborder" valign="top" width="21.060000000000002%" headers="mcps1.2.4.1.1 "><p id="p1359055518568"><a name="p1359055518568"></a><a name="p1359055518568"></a>dit_fsdp</p>
</td>
<td class="cellrowborder" valign="top" width="18.93%" headers="mcps1.2.4.1.2 "><p id="p1759010553565"><a name="p1759010553565"></a><a name="p1759010553565"></a>Fully Sharded Data Parallel (FSDP) for DiT </p>
</td>
<td class="cellrowborder" valign="top" width="60.01%" headers="mcps1.2.4.1.3 "><p id="p12590055185611"><a name="p12590055185611"></a><a name="p12590055185611"></a>Whether the DiT model uses the FSDP policy.</p>
</td>
</tr>
<tr id="row431618018575"><td class="cellrowborder" valign="top" width="21.060000000000002%" headers="mcps1.2.4.1.1 "><p id="p153177019575"><a name="p153177019575"></a><a name="p153177019575"></a>t5_fsdp</p>
</td>
<td class="cellrowborder" valign="top" width="18.93%" headers="mcps1.2.4.1.2 "><p id="p33174018573"><a name="p33174018573"></a><a name="p33174018573"></a>FSDP for Text-To-Text Transfer Transformer (T5) </p>
</td>
<td class="cellrowborder" valign="top" width="60.01%" headers="mcps1.2.4.1.3 "><p id="p9317301573"><a name="p9317301573"></a><a name="p9317301573"></a>Whether the T5 model uses the FSDP policy.</p>
</td>
</tr>
<tr id="row11402154312018"><td class="cellrowborder" valign="top" width="21.060000000000002%" headers="mcps1.2.4.1.1 "><p id="p194039438019"><a name="p194039438019"></a><a name="p194039438019"></a>vae_parallel</p>
</td>
<td class="cellrowborder" valign="top" width="18.93%" headers="mcps1.2.4.1.2 "><p id="p24036431804"><a name="p24036431804"></a><a name="p24036431804"></a>Whether to enable VAE parallel policy</p>
</td>
<td class="cellrowborder" valign="top" width="60.01%" headers="mcps1.2.4.1.3 "><p id="p1940334314013"><a name="p1940334314013"></a><a name="p1940334314013"></a>Whether the VAE model uses the parallel policy.</p>
</td>
</tr>
</tbody>
</table>
