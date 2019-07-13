function makeShader(type, gl, source) {
  var shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    console.log(gl.getShaderInfoLog(shader));
  }
  return shader;
}

function makeVertexShader(gl, source) {
  return makeShader(gl.VERTEX_SHADER, gl, source);
}

function makeFragmentShader(gl, source) {
  return makeShader(gl.FRAGMENT_SHADER, gl, source);
}

function makeTexture(gl, index, source, options) {
  if (options == undefined) options = {};
  if (options.target === undefined) options.target = gl.TEXTURE_2D;
  if (options.level === undefined) options.level = 0;
  if (options.internalFormat === undefined)
    options.internalFormat = gl.RGB;
  if (options.border === undefined) options.border = 0;
  if (options.format === undefined) options.format = gl.RGB;
  if (options.type === undefined) options.type = gl.UNSIGNED_BYTE;
  if (options.wrapS === undefined) options.wrapS = gl.CLAMP_TO_EDGE;
  if (options.wrapT === undefined) options.wrapT = gl.CLAMP_TO_EDGE;
  if (options.minFilter === undefined) options.minFilter = gl.LINEAR;
  gl.activeTexture(gl.TEXTURE0 + index);
  gl.bindTexture(gl.TEXTURE_2D, gl.createTexture());
  if (options.width === undefined && options.height === undefined) {
    gl.texImage2D(
      options.target, options.level, options.internalFormat,
      options.format, options.type, source
    );
  } else {
    gl.texImage2D(
      options.target, options.level, options.internalFormat,
      options.width, options.height, options.border,
      options.format, options.type, source
    );
  }
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, options.wrapS);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, options.wrapT);
  gl.texParameteri(
    gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, options.minFilter
  );
  return index;
}
