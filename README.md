# Application of NerfMM to fNIRS

This project aims to cluster (Green) bright dots on top of a quarter cycle video. 

The project report is in the pdf file located inside this directory.

## Installation

Clone the project - 

`git clone https://github.com/ventusff/improved-nerfmm`

Install the dependencies - Same as [improved-nerfmm](https://github.com/ventusff/improved-nerfmm).

## Usage

This adds one file to the improved-nerfmm implementation - clustering.py is meant to cluster the dots, and uses the same dir arguments as the regular python files. Only 1 parameter is typically required.

In addition, we added another three training flags:
* `model_init` which get a path to the model we want to initial our scene on.
* `cam_init` which is the same as the former, but for the camera's parameters.
* `cam_init_ours` which initialized the camera to be in a position of an arch, in the shape of a quarter of a circle. This position aims to mimic the original camera path.

## Results

We organized our results in the following table:

|Ground Truth |Initialized From | Learning Rate Camera |Learning Rate Scene | Renders | Camera Trajectory | Loss Graph |
|:---: | :---: | :---: | :---: | :---: |:---: |:---:| 
|[GX011613](https://drive.google.com/file/d/1ulNQahD-XdZEx-0GndawxKAGZPM2eeAR/view) | [GX011625](https://drive.google.com/file/d/1excRv40d2u8MxgtwOj_pkqoiyy5fNVHT/view) | 1.00E-03 | 1.00E-03 | [exp1](https://drive.google.com/drive/u/1/folders/1kX-JNhWgoDnpH3QP6pS4cC17VBKTj1aF) | [exp1](https://drive.google.com/file/d/1pYpUC9S6IEXUPLs-4qfdOjpPU6g5OtuK/view) | TBP|
|[GX011613](https://drive.google.com/file/d/1ulNQahD-XdZEx-0GndawxKAGZPM2eeAR/view) | [GX011625](https://drive.google.com/file/d/1excRv40d2u8MxgtwOj_pkqoiyy5fNVHT/view) | 5.00E-04 | 1.00E-04 | [exp2](https://drive.google.com/drive/u/1/folders/1wmn8atCZtluj5quWXSFQEb0rt08Jh7pE) | [exp2](https://drive.google.com/file/d/1NfvPP18mrTo-xhfe11q4v9HKI4zMnkwj/view) | [exp2](https://drive.google.com/file/d/168f-WBZfsmSs2ZK4cf2g12d7HZ5gXOht/view)|
|[GX011613](https://drive.google.com/file/d/1ulNQahD-XdZEx-0GndawxKAGZPM2eeAR/view) | [GX011625](https://drive.google.com/file/d/1excRv40d2u8MxgtwOj_pkqoiyy5fNVHT/view) |1.00E-05283 | 1.00E-05 | [exp3](https://drive.google.com/drive/u/1/folders/1j17BE-VwQCxM0uSyMR2Be3OQG0aYoBVY) | [exp3](https://drive.google.com/file/d/1sT_O4CrlR4YF7tAC-pUeVZTGG1yEiDYK/view) | [exp3](https://drive.google.com/file/d/10sekH-wjhiOofv01gniON7etVmecZ154/view)|
|[GX011520](https://drive.google.com/file/d/1ziEb4HXgfWon9mN_6wjQIygrRxKzjefC/view) | [GX011625](https://drive.google.com/file/d/1excRv40d2u8MxgtwOj_pkqoiyy5fNVHT/view) | 1.00E-03 | 1.00E-03 | [exp4](https://drive.google.com/drive/u/1/folders/1H7rdQYEymhGqxxFCvkk5QYNqSmVHQFEj) | [exp4](https://drive.google.com/file/d/13wHYuQRWO4Ivn9sxiHVXvodMHP4Dt_41/view) | [exp4](https://drive.google.com/file/d/1UKAR_MjvdyZEi7KM33VAc2NeuO3bHcPc/view) |
|[GX011520](https://drive.google.com/file/d/1ziEb4HXgfWon9mN_6wjQIygrRxKzjefC/view) | [GX011625](https://drive.google.com/file/d/1excRv40d2u8MxgtwOj_pkqoiyy5fNVHT/view) | 5.00E-04 | 1.00E-04 | [exp5](https://drive.google.com/drive/u/1/folders/1lXnR7DzReecEhQSGN1YJ81H9kN9GGWap) | [exp5](https://drive.google.com/file/d/1suc1ZKWcfYVJFIovkvMBgHjAV0b1Kfy5/view) | [exp5](https://drive.google.com/file/d/1HnhhfBUa143drCnUhcQzrrATe-JSsmkq/view) |
|[GX011520](https://drive.google.com/file/d/1ziEb4HXgfWon9mN_6wjQIygrRxKzjefC/view) | [GX011625](https://drive.google.com/file/d/1excRv40d2u8MxgtwOj_pkqoiyy5fNVHT/view) | 1.00E-05 | 1.00E-05 | [exp6](https://drive.google.com/drive/u/1/folders/1dA-2lN592ll5xmIHzXFc0nAkk58LK9sK) | [exp6](https://drive.google.com/file/d/1JN7gUNOaYwjO0rdqaOWaB3EtwbJ5Py8Z/view) | [exp6](https://drive.google.com/file/d/1IxiMh5DyFzcIV4BDOdqBSsLitKgbrEt3/view) |
|[GX011625](https://drive.google.com/file/d/1excRv40d2u8MxgtwOj_pkqoiyy5fNVHT/view) | Our Camera Path | 5.00E-04 | 1.00E-03 | [exp7](https://drive.google.com/drive/u/1/folders/1Cpc9MUOIHGQcNtJczSByxhV_HI959GHn) | [exp7](https://drive.google.com/file/d/1gTKkx-IoNXcHJrAtNyIVpIShfyeRoz8C/view) | [exp7](https://drive.google.com/file/d/1-3sR6C4lagIjaq1crkU4Qjvh2zrN9SlY/view) |
|[GX011613](https://drive.google.com/file/d/1ulNQahD-XdZEx-0GndawxKAGZPM2eeAR/view) | Our Camera Path | 5.00E-04 | 1.00E-03 | [exp8](https://drive.google.com/drive/u/1/folders/1_HXkTEQGsrRS4nrd9O8d28A-Hgl4UE1S) | [exp8](https://drive.google.com/file/d/1AbUjWku1IMpkIiIQL8lIstOlbOsmYAlv/view) | [exp8](https://drive.google.com/file/d/1ITQQBZrU5hwn6lakNzEffEgKjRQSBHaH/view) |
|[GX011520](https://drive.google.com/file/d/1ziEb4HXgfWon9mN_6wjQIygrRxKzjefC/view) | Our Camera Path | 5.00E-04 | 1.00E-03 | [exp9](https://drive.google.com/drive/u/1/folders/1_HXkTEQGsrRS4nrd9O8d28A-Hgl4UE1S) | [exp9](https://drive.google.com/file/d/1oBtMUSHcz4VW1d6kJZ_Zw33y8q52Thqb/view) | [exp9](https://drive.google.com/file/d/1PZK8IYX3W0_k9AaeVbZXEgFDEqmF2ttM/view) |
|[GX011613](https://drive.google.com/file/d/1ulNQahD-XdZEx-0GndawxKAGZPM2eeAR/view) | [GX011625 Camera](https://drive.google.com/file/d/1excRv40d2u8MxgtwOj_pkqoiyy5fNVHT/view) | 5.00E-04 | 1.00E-03 | [exp10](https://drive.google.com/drive/u/1/folders/1rBVqiVHL6Iaw9Gj9klG4XRDosO0V37jw) | [exp10](https://drive.google.com/file/d/1_mpw5gt-j3GqR2B023zXWAXQLkw3SGNP/view) | [exp10](https://drive.google.com/file/d/1pmRnosRkJO_U-CsxShYRBsl-Rf_ItN1H/view) |
|[GX011520](https://drive.google.com/file/d/1ziEb4HXgfWon9mN_6wjQIygrRxKzjefC/view) | [GX011625 Camera](https://drive.google.com/file/d/1excRv40d2u8MxgtwOj_pkqoiyy5fNVHT/view) | 5.00E-04 | 1.00E-03 | [exp11](https://drive.google.com/drive/u/1/folders/1wuGhaX5yE4pD0TNH_KO6tTOqQGCrhZ4V) | [exp11](https://drive.google.com/file/d/1JCKFqoW8VzOuRy6Su8mrUYAVcQUuTk-i/view) | [exp11](https://drive.google.com/file/d/1mO0CkLMRwUMCMNu07DgUUR7QDgL74526/view) |
|[GX011520](https://drive.google.com/file/d/1ziEb4HXgfWon9mN_6wjQIygrRxKzjefC/view) |  | 1.00E-03 | 1.00E-03 | [exp12](https://drive.google.com/drive/u/1/folders/1ce0ob6Zj3sEZRpitdFzXdjdEt14LoUnD) | [exp12](https://drive.google.com/file/d/17TciDnpIhRclKi31Cus3AAcbONlK_cS5/view) | [exp12](https://drive.google.com/file/d/1VjzL3mr9GKkMjyHpR4UyZdA9XrFPVsKA/view) |
|[GX011613](https://drive.google.com/file/d/1ulNQahD-XdZEx-0GndawxKAGZPM2eeAR/view) |  | 1.00E-03 | 1.00E-03 | [exp13](https://drive.google.com/drive/u/1/folders/1_VxDj7J0qWWm0i5jC-knnH-Fv_p4JODJ) | [exp13](https://drive.google.com/file/d/1CiOcEgmMR1fnEGY8cZK_QW3PVRJ64E1A/view) | [exp13](http://drive.google.com/file/d/1bMjRCZWk2p37w1tztUgVCrtZO0JhtDSl/view) |
|[GX011625](https://drive.google.com/file/d/1excRv40d2u8MxgtwOj_pkqoiyy5fNVHT/view) |  | 1.00E-03 | 1.00E-03 | [exp14](https://drive.google.com/drive/u/1/folders/1qcxgT9cC4nnKk44okCogINTIssMESkYX) | [exp14](https://drive.google.com/file/d/1HwjPbgUoz_GVXXvWzDBVkDDqgZuOVQtw/view) | [exp14](https://drive.google.com/file/d/18UeyQ7h018rJ9I18i179LVGD-rHU4ycM/view) |
|[GX011625](https://drive.google.com/file/d/1excRv40d2u8MxgtwOj_pkqoiyy5fNVHT/view) |  | 1.00E-03 | 1.00E-03 | [Not Siren](https://drive.google.com/drive/u/1/folders/1a0yIXCm__VDUXvpnLzR5VbFWbM2VCyW4) | [Not Siren](https://drive.google.com/file/d/1IZxdl6FuQUgHK484ONfWSWsRhH6sblT-/view) | [Not Siren](https://drive.google.com/file/d/1DMRwiyyGZi3TdfehowLlgkN7L4Q0MvGT/view) |


## Authors
This was made by Amit Sandler and Tomer Porian, for a workshop in Computational Graphics.

## Contributing
Help
