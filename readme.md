# AI experiments with PWA-PolymeR frontend and Java-Spring-Tensorflow backend

This application demonstrates how to integrate TensorFlow with you Progressive Web Application. The idea is to use different pre-trained deep learning models.

## Release 0.1 (pre-alpha)
An initial release with PWA front-end and Java+Spring+Tensorflow backend. Using pre-trained imagenet_comp_graph model (https://github.com/miyosuda/TensorFlowAndroidDemo/tree/master/app/src/main/assets) with 1001 different image classed.

<strong>Caution: There is some issue still with recognizer engine, that will be fixed in the next release.</strong>
 
## Requirements

- Java 1.8
- Maven
- NPM
- Bower

Before you start, be sure you have Node installed. If you are on Mac, I suggest using [Homebrew](http://brew.sh/) to install it. 
When you have node installed, install bower:
 
 ```npm install -g bower```
 
## Running

You can start the application locally by running:
 
```mvn package spring-boot:run```

Open [http://localhost:5000](http://localhost:5000) in your browser

# Production

For a production optimized version, run with the `production` profile. This will run vulcanize on your HTML imports to bundle them all into one file.

```mvn clean package spring-boot:run -Pproduction```

## Resources for Polymer development

- [Polymer Homepage](https://www.polymer-project.org/) - catalog of Polymer elements and guides
- [Vaadin Elements](https://vaadin.com/elements) - open source web components for business oriented apps.

## References 

- [Spring boot PWA example](https://github.com/vaadin-marcus/polymer-spring-boot) - Spring boot PWA example by Marcus Hellberg, this example is used as a basis of this ai-experiments project.
