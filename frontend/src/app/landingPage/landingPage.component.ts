// login.component.ts
import { Router } from '@angular/router';
import { Component } from '@angular/core';

@Component({
  selector: 'app-test',
  templateUrl: './landingPage.component.html',
  styleUrls: ['./landingPage.component.css']
})
export class LandingPageComponent {
  constructor( private router: Router) {}

  email: any
  pass: any

  

}
