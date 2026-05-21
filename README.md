# GaussianSplatRenderingDeinoiser

* 2 key assumptions - static scene, only camera movement + only using post processing effects for cleanup using the outputted stochastic color and depth texture



# setup

cd GaussianSplatRenderingDeinoiser
npm install
npm run dev


# notes
- l1 dist - doesnt overpunish errors, prevents image blurring by the model
- recurrent hidden states - encode previous frames to remove random per frame noise
- dataset construction
    - sample 20 frames, pause for 10, move camera, repeat -  still frames provide information about global structure to the model so it can differentiate noise
    - pass in sequences of frames to build recurrent hidden states

- potentially try charbonnier_loss