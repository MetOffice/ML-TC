# Contributing Guidelines

You'd like to help? Great!  :tada:

[Clone your own local copy](https://help.github.com/en/articles/cloning-a-repository) of this repositry run the following in your terminal:

```shell
git clone https://github.com/MetOffice/ML-TC.git
```

Consider [creating a conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) from the dependencies specified in the `environment.yml` file:
```shell
conda env create -f environment.yml
```
Remember to activate your new environment:
```shell
conda activate ml-tc
```

:exclamation: *Note: We are currently only able to provide some of the original model data used to train the models in this repo, principally due to the data volume. See [Data](https://github.com/MetOffice/ML-TC/blob/initial-setup/README.md#data) for some data sources.

# Before you start...
Read through the current issues to see what you can help with.  If you have your own ideas for improvements, please start a new issues so we can track and discuss your improvement. You must create a new branch for any changes you make.

**Please take note of the following guidelines when contributing to the ML-TC repository.**

* Please do **not** make changes to the `main` branch.  The `main` branch is reserved for files and code that has been fully tested and reviewed.  Only the core ML-TC developers can/should push to the `main` branch.

* The `develop` branch contains the latest holistic version of the `ML-TC` repository.  Please branch off `develop` to fix a particular issue or explore a new idea.

* Please use the following tokens at the start of a new branch name to help sign-post and group branches:

Name | Description
---- | -----------
new | Branch adding new code/files that don't exist in the repo
fix | Branch modifying code/files that already exist in the repo.
junk | Throwaway branch created to experiment

* Git can pattern match branches to to give you an overview of all (e.g. fix) branches:
 ```shell
 git branch --list "fix/*"
 ```
* Use a forward slash to separate the token from the branch name. For example:
```
new/cgan-4
fix/get-training-data
```
* When you think your branch is ready to be merged into `develop`, open a new pull request.

## Signposting
* **Issues** are tracked and discussed under the Issues tab.  Please use issues to disucss proposed changes or capture improvements needed to work towards the next milestone.  Issues or improvements that contribute to the next milestone to be captured in thr Wiki tab.
* **Pull requests** show branches that are currently under review.  New pull requests are created in reponse to branch fixes identified and recorded in the Issues tab.
* **Projects** contains kanban-style boards for summarising update aims for future versions of the notebooks, and to record speculative improvements that cannot be action in the current milestone.



Other more general points to note:

* **Avoid long descriptive names.**  Long branch names can be very helpful when you are looking at a list of branches but it can get in the way when looking at decorated one-line logs as the branch names can eat up most of the single line and abbreviate the visible part of the log.
* **Do not use bare numbers.** Do not use use bare numbers (or hex numbers) as part of your branch naming scheme.


<h5 align="center">
<img src="etc/MO_MASTER_black_mono_for_light_backg_RBG.png" width="200" alt="Met Office"> <br>
&copy; British Crown Copyright, Met Office
</h5>
