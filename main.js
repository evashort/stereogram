var image;

function init() {
  var canvas = document.createElement("canvas");
  canvas.width = 600;
  canvas.height = 400;
  document.body.appendChild(canvas);
  gl = canvas.getContext("webgl");

  var shaderProgram = gl.createProgram();
  gl.attachShader(shaderProgram, makeVertexShader(gl, `
    attribute vec2 vertexPosition;

    varying highp vec2 position;

    void main(void) {
      gl_Position = vec4(
        vec2(2.0, -2.0) * vertexPosition + vec2(-1.0, 1.0),
        0.0,
        1.0
      );
      position = vertexPosition;
    }
  `));
  gl.attachShader(shaderProgram, makeFragmentShader(gl, `
    precision highp float;
    varying vec2 position;

    uniform sampler2D image;
    uniform sampler2D offsetMap;

    void main(void) {
      float offset = texture2D(offsetMap, position).a;
      vec2 newPosition = position + vec2(0.0, offset);
      gl_FragColor = texture2D(image, newPosition);
    }
  `));
  gl.linkProgram(shaderProgram);
  gl.useProgram(shaderProgram);

  gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
    0.0,  0.0,
    0.0,  1.0,
    1.0,  0.0,
    1.0,  1.0
  ]), gl.STATIC_DRAW);
  gl.vertexAttribPointer(
      gl.getAttribLocation(shaderProgram, 'vertexPosition'),
      2, // numComponents
      gl.FLOAT, // type
      false, // normalize
      0, // stride
      0 // offset
  );
  gl.enableVertexAttribArray(
    gl.getAttribLocation(shaderProgram, 'vertexPosition')
  );

  image = document.getElementsByTagName("img")[0];
  gl.uniform1i(
    gl.getUniformLocation(shaderProgram, 'image'),
    makeTexture(gl, 0, image)
  );

  gl.getExtension("OES_texture_float");
  gl.getExtension("OES_texture_float_linear");

  offsetMap = new Float32Array(canvas.width * canvas.height);
  for (var x = 170; x < 250; x++) {
    for (var y = 50; y < 185; y++) {
      offsetMap[y * canvas.width + x] = 0.01;
    }
  }
  gl.uniform1i(
    gl.getUniformLocation(shaderProgram, 'offsetMap'),
    makeTexture(
      gl, 1, offsetMap, {
        internalFormat: gl.ALPHA,
        width: canvas.width,
        height: canvas.height,
        format: gl.ALPHA,
        type: gl.FLOAT
      }
    )
  );

  gl.drawArrays(
    gl.TRIANGLE_STRIP,
    0, // first
    4, // count
  );
}
