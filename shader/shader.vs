#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;   // <- 新增

out vec3 vColor;                       // 传给 fragment

uniform mat4 uMVP;

void main(){
    gl_Position = uMVP * vec4(aPos, 1.0);
    vColor = aColor;                   // 传递下去
    gl_PointSize = 5.0;
}
