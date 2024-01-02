const src = '';
const byteCharacters = atob(src);
const byteNumbers = new Array(byteCharacters.length);
for (let i = 0; i < byteCharacters.length; i++) {
    byteNumbers[i] = byteCharacters.charCodeAt(i);
}
const byteArray = new Uint8Array(byteNumbers);
const blob = new Blob([byteArray], {type: 'image/jpeg'});
var imageUrl = URL.createObjectURL(blob);
// document.querySelector("#image").src = imageUrl;
console.log(imageUrl)