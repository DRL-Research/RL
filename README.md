<a name="readme-top"></a>



<h1 align="center">Novel distributed learning-based approach for joint navigation in local environments</h1>
  <p align="center">
    project_description
    <br />
    <br />
    <a href="https://github.com/github_username/repo_name">View Demo</a>
    ·
    <a href="https://github.com/github_username/repo_name/issues">Report Bug</a>
    ·
    <a href="https://github.com/github_username/repo_name/issues">Request Feature</a>
  </p>



## About The Project

...




<br>
<br>

<h3 style="color:green;">Save Experiments logs</h3>

In <b>main.py</b> change the log directory 

    log_directory = "exp2"

In this folder the rewards and losses will be saved



<br>
<br>

<h3 style="color:green;">View logs in TensorBoard</h3>

Activate TensorBoard in Terminal, in pycharm (pay attention to directory =exp2/):

    python -m tensorboard.main --logdir=exp2/

Go to the following URL in the browser:

    http://localhost:6006/


<br>
<br>


<h3 style="color:green;"> How to Speedup the simulation:</h3>

1. Unreal - downfacing arrow by the Play button - Advanced Settings - 
   Search for "Use less CPU in the background" and disable it (tick is on)
2. Unreal - Settings - Engine Scalability Settings - Low
3. Unreal - Shrink Simulation window inside Unreal Editor as much as possible so the simulation will consume fewer resources
   (thus crashes should be prevented) 
4. AirSim settings.json - Replace your file with the one in the git repo  (settings - original.json kept for reference)
   If the Unreal Editor Crashes - Decrease ClockSpeed in settings.json - max possible value depends on your HW specs).
   Ido stabilized the simulation at ClockSpeed = 3, but don't be afraid to explore higher speeds.
   If you do not have an Nvidia GPU, delete the blocks starting with GpuId and UseNvidiaHardwareEncoder.


<br>
<br>

<h3 style="color:green;">Change settings of airsim</h3>

change the json settings file in the path:

    This PC/Documents/AirSim/settings.json


<br>
<br>
<h3 style="color:green;">Commit to GitHub</h3>

when offered to commit and push, pay attention to changing:

    master -> origin : master

To:

    master -> origin : main


<p align="right">(<a href="#readme-top">back to top</a>)</p>



## Experiments:

### Exp1:

Car2 coming from LTR 3 times, and then from RTL 3 times,
Then once again the same thing (3 LTR followed by 3 RTL).
Each time the model keeps training on prev version.

| Parameters     | Values                          | Notes                       |
|----------------|---------------------------------|-----------------------------|
| Location Car1  | [-20,0]                         |                             |
| Location Car2  | [0,-10] / [0,10]                | Car2 coming from RTL or LTR |
| Learning rate  | 0.003                           |                             |
| Local / Global | Local in Car1, no global at all |                             |












<br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br>
<br><br><br><br><br><br><br>

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* npm
  ```sh
  npm install npm@latest -g
  ```

### Installation

1. Get a free API Key at [https://example.com](https://example.com)
2. Clone the repo
   ```sh
   git clone https://github.com/github_username/repo_name.git
   ```
3. Install NPM packages
   ```sh
   npm install
   ```
4. Enter your API in `config.js`
   ```js
   const API_KEY = 'ENTER YOUR API';
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [ ] Feature 1
- [ ] Feature 2
- [ ] Feature 3
    - [ ] Nested Feature

See the [open issues](https://github.com/github_username/repo_name/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Your Name - [@twitter_handle](https://twitter.com/twitter_handle) - email@email_client.com

Project Link: [https://github.com/github_username/repo_name](https://github.com/github_username/repo_name)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* []()
* []()
* []()

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo_name.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo_name/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo_name.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo_name/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo_name.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo_name/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo_name.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo_name/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo_name.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo_name/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 
