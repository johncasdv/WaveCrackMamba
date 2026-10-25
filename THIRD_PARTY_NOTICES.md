# Third-Party Notices

## DySample

The `DySample` implementation in
`wavecrack_mamba_R1a_code_20261022/wavecrack_mamba_R1a_code_20261022/projects/wavecrack_mamba/models/inverse_pyramid.py`
is adapted from [tiny-smart/dysample](https://github.com/tiny-smart/dysample).
The reported R1a model uses fixed bilinear upsampling, but the DySample
implementation is included for the R1b configuration.

The upstream work is licensed under the MIT License:

```text
MIT License

Copyright (c) 2023 Wenze Liu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

The project's external Python dependencies are installed separately and remain
under their respective licenses. Dataset images and masks and the public
ConvNeXt pretraining checkpoint are not distributed in this repository.
